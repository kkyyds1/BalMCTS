import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from gnn.ExperienceReplay import ExperienceReplayBuffer
from entry.data import *
# class DataProcessor:
#     def __init__(self, variables, constraints):
#         self.variables = variables
#         self.constraints = constraints

#     def get_variable_info(self):
#         variable_info = []
#         for variable in self.variables:
#             domain_length = len(self.variables[variable].domain)
#             is_assigned = self.variables[variable].is_assigned
#             variable_info.append([domain_length, is_assigned])
#         return variable_info

#     def get_constraint_info(self):
#         constraint_info = []
#         domain_product = 1
#         for constraint in self.constraints:
#             for var in self.constraints[constraint].variables:
#                 domain_product *= len(self.variables[var].domain)
#             allowed_tuples = len(self.constraints[constraint].relations)
#             num_variables = len(self.constraints[constraint].variables)
#             dynamic_compactness = 1 - allowed_tuples / domain_product
#             constraint_info.append([num_variables, dynamic_compactness])
#         return constraint_info

#     def generate_edge_index(self):
#         edge_index = []
#         var_constr_index = [[] for _ in range(len(self.variables))]
#         constr_var_index = [[] for _ in range(len(self.constraints))]
#         for constraint in self.constraints.values():
#             constr_index = constraint.index 
            
#             for var in constraint.variables:
#                 var_index = self.variables[var].index
#                 var_constr_index[var_index].append(constr_index)
#                 constr_var_index[constr_index].append(var_index)
               
#                 edge_index.append([constr_index + len(self.variables), var_index])
#                 edge_index.append([var_index, constr_index + len(self.variables)])

#         return edge_index, var_constr_index, constr_var_index

#     def create_data(self):
#         variable_info = self.get_variable_info()
#         constraint_info = self.get_constraint_info()

#         edge_index, var_constr_index, constr_var_index = self.generate_edge_index()
#         data = Data(x=None, edge_index=edge_index)
#         data.x = torch.tensor(variable_info + constraint_info, dtype=torch.float)
#         data.num_nodes = len(var_constr_index) + len(constr_var_index)
        
#         return data, var_constr_index, constr_var_index


def get_variable_info(variables):
    variable_info = []
    for variable in variables:
        domain_length = len(variables[variable].domain)
        is_assigned = variables[variable].is_assigned
        variable_info.append([domain_length, is_assigned])
    return variable_info

def get_constraint_info(variables, constraints):
    constraint_info = []
    domain_product = 1
    for constraint in constraints:
        for var in constraints[constraint].variables:
            domain_product *= len(variables[var].domain)
        allowed_tuples = len(constraints[constraint].relations)
        # domain_product = len(dom1) * len(dom2)
        num_variables = len(constraints[constraint].variables)
        dynamic_compactness = 1 - allowed_tuples / domain_product
        constraint_info.append([num_variables, dynamic_compactness])
    return constraint_info

def generate_edge_index(variables, constraints):
    edge_index = []
    var_constr_index = [[] for _ in range(len(variables))]
    constr_var_index = [[] for _ in range(len(constraints))]
    for constraint in constraints.values():
        constr_index = constraint.index 
        
        for var in constraint.variables:
            var_index = variables[var].index
            var_constr_index[var_index].append(constr_index)
            constr_var_index[constr_index].append(var_index)
           
            edge_index.append([constr_index + + len(variables), var_index])
            edge_index.append([var_index, constr_index + + len(variables)])

    return edge_index, var_constr_index, constr_var_index

def create_data(variables, constraints):
    variable_info = get_variable_info(variables)
    constraint_info = get_constraint_info(variables, constraints)

    edge_index, var_constr_index, constr_var_index = generate_edge_index(variables, constraints)
    # 创建 Data 对象
    data = Data(x=None, edge_index=edge_index)

    # 设置节点和边的属性
    data.x = torch.tensor(variable_info + constraint_info, dtype=torch.float)
    data.num_nodes = len(var_constr_index) + len(constr_var_index)
    
    return data, var_constr_index, constr_var_index

class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, init_input_dim, init_output_dim, num_layers):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        
        self.fc_variable   = nn.Linear(input_dim, output_dim)
        self.fc_constraint = nn.Linear(input_dim, output_dim)
        
        self.fc_init_variables = nn.Linear(init_input_dim, init_output_dim)
        self.fc_init_constraints = nn.Linear(init_input_dim, init_output_dim)
        
        self.fc_q_value = nn.Linear(output_dim * 2, 1)
    
    def initialize_layer_zero(self, x, num_variables):
        variable_features = x[:num_variables]
        constraint_features = x[num_variables:]

        raw_variable_features = self.fc_init_variables(variable_features)
        raw_constraint_features = self.fc_init_constraints(constraint_features)

        self.last_variable_features  = raw_variable_features
        self.last_constraint_features = raw_constraint_features

    def aggregate(self, var_constr_index, constr_var_index, x):
        # 获取变量节点和约束节点的初始特征
        variable_features = x[:len(var_constr_index)]
        constraint_features = x[len(var_constr_index):]
        zeros_tensor = torch.zeros((1, self.output_dim))

        for k in range(1, self.num_layers):
            last_variable_features  = torch.cat((self.last_variable_features, zeros_tensor))
            last_constraint_features = torch.cat((self.last_constraint_features, zeros_tensor))
           
            # 获取约束节点的邻居变量节点的索引
            constraint_neighbor_indices = constr_var_index
            # 填充子列表
            max_length = max(len(indices) for indices in constraint_neighbor_indices)
            constraint_neighbor_indices = [indices + [-1] * (max_length - len(indices)) for indices in constraint_neighbor_indices]
            
            # 获取邻居变量节点的特征
            constraint_neighbor_features = last_variable_features[[constraint_neighbor_indices]]
            
            # 对邻居变量节点的特征进行求和
            constraint_aggregated_features = torch.sum(constraint_neighbor_features, dim=1)

            # 拼接特征矩阵
            constraint_cat_features = torch.cat([constraint_aggregated_features, self.last_constraint_features, constraint_features], dim=1)
            
            self.last_constraint_features = self.fc_constraint(constraint_cat_features)

            # 获取约束节点的邻居变量节点的索引
            variable_neighbor_indices = var_constr_index
            # 填充自列表
            max_length = max(len(indices) for indices in variable_neighbor_indices)
            variable_neighbor_indices = [indices + [-1] * (max_length - len(indices)) for indices in variable_neighbor_indices]
            
            # 获取邻居变量节点的特征
            variable_neighbor_features = last_constraint_features[[variable_neighbor_indices]]
           
            # 对邻居变量节点信息聚合
            variable_aggregated_features = torch.sum(variable_neighbor_features, dim=1)
            
            # 拼接特征矩阵
            variable_cat_features = torch.cat([variable_aggregated_features, self.last_variable_features, variable_features], dim=1)
            
            self.last_variable_features = self.fc_variable(variable_cat_features)

    def embedding_Q(self):
        variable_aggregated_features = torch.sum(self.last_variable_features, dim = 0)
        variable_aggregated_features = variable_aggregated_features.repeat(len(self.last_variable_features), 1)
        variable_cat_features = torch.cat([variable_aggregated_features, self.last_variable_features], dim=1)
        Q = self.fc_q_value(variable_cat_features)
        return Q
    
    def predict(self, var_constr_index, constr_var_index, data, action=False):
        # 创建变量和约束每一层的特征张量
        self.initialize_layer_zero(data.x, len(var_constr_index))
        self.aggregate(var_constr_index, constr_var_index, data.x)
        
        Q = self.embedding_Q(data,var_constr_index)
        if action:
            for var_index in range(len(var_constr_index)):
                if int(data.x[var_index][1]):
                    Q[var_index] = float('inf')
            return torch.argmin(Q).item()
        return Q
    
class DDQN:
    def __init__(self, input_dim, output_dim, init_input_dim, init_output_dim, num_layers, phase, path=None):
        self.target_net = GNN(input_dim, output_dim, init_input_dim, init_output_dim, num_layers)
        self.online_net = GNN(input_dim, output_dim, init_input_dim, init_output_dim, num_layers)
        self.phase = phase
        self.experience_buffer = ExperienceReplayBuffer(500)
        self.losses = []
        if phase == 'Training':
            self.load_model(path)

    def samples(self, epr, batch_size):
        samples = epr.sample(batch_size)
        losses = []
        for sample in samples:
            state, var_constr_index, constr_var_index, action, next_state, next_var_constr_index, next_constr_var_index, reward, T = sample
            
            predicted_q = self.online_net.predict(var_constr_index, constr_var_index, state, action=True)  # Select the Q value for the action and add a dimension
            
            predicted_action = self.online_net.predict(next_var_constr_index, next_constr_var_index, next_state, action=True)
            
            # 如果是终止状态，目标Q值就是奖励
            if T:
                y = torch.tensor([reward])
            else:
                # 否则，目标Q值是奖励加上折扣后的未来最小Q值
                next_Q = self.target_net.predict(next_var_constr_index, next_constr_var_index, next_state)
                next_q = next_Q(predicted_action)
                y = torch.tensor([reward]) + 0.99 * next_q
            loss = self.online_net.update_parameters(predicted_q, y)
            
            losses.append(loss.item())

        return losses

    def update_model(self, old_net, new_net):
        new_state_dict = self.online_net.state_dict()
        self.target_net.load_state_dict(new_state_dict)

    def save_model(self, path):
        torch.save(self.target.state_dict(), path)
    
    def load_model(self, path):
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        
    def update_parameters(self, predicted_q, target_q):
        optimizer = optim.Adam(self.online_net.parameters(), lr=0.01)
        # Compute the loss
        loss = F.mse_loss(predicted_q, target_q)

        # Zero gradients
        optimizer.zero_grad()

        # Perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

        return loss

    def store_experience(self, assignment, variables, local_assignment, update_variables, constraints, action, count, train_freq, update_freq):
        if self.phase == 'Training':
            state, var_constr_index, constr_var_index = create_data(variables, constraints)
            action_index = int(action[1:])
            next_state, next_var_constr_index, next_constr_var_index = create_data(update_variables, constraints)
            
            # 把经验放到缓冲区
            if len(local_assignment) == len(variables):
                self.experience_buffer.add((state, var_constr_index, constr_var_index, action_index, next_state, next_var_constr_index, next_constr_var_index, 1, 1))
            else:
                self.experience_buffer.add((state, var_constr_index, constr_var_index, action_index, next_state, next_var_constr_index, next_constr_var_index, 1, 0))
            
            # nstep采样， 更新online net
            if count % train_freq == 0:
                loss = self.samples(epr, batch_size=train_freq)
                self.losses.extend(loss)

            # 把online net 的参数复制给target net
            if count % update_freq == 0:
                self.update_model()
# data = create_data(variables, constraints)

# # 创建 GNN 模型
# gnn = GNN(input_dim=12, output_dim=5, init_input_dim = 2, init_output_dim=5, num_layers=3, data=data)
# action = gnn.predict(variables, constraints)
