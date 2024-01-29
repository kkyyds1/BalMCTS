import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from dataclasses import dataclass
from ast import literal_eval


@dataclass
class Variable:
    index: int
    domain: list
    is_assigned: int
    constraints: list

@dataclass
class Constraint:
    index: int
    variables: tuple
    relations: list
    product: int

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# variables, constraints = parser_data(f'output/problems/{0}.txt')
def parser_data(filepath):
    variables = {}
    constraints = {}

    caseFile=open(filepath,"r")
    lines=caseFile.readlines()

    # Read head
    paramValues=lines[1].split('\t')
    n = int(paramValues[0])
    d = int(paramValues[1])
    m = int(paramValues[2])
    k = int(paramValues[3])
    nb = int(paramValues[4])

    for i in range(n):
        var = Variable(i, list(range(d)), 0, [])
        variables[f'x{i}'] = var
    
    constr_var = {}
    index = 0
    for i in range(m):
        
        constrStrings=lines[i+3].split('|')
        
        var1, var2 = literal_eval(constrStrings[0])
        
        allowedTuples = literal_eval(constrStrings[1])
        
        domain_product = len(variables[f'x{var1}'].domain) * len(variables[f'x{var2}'].domain)
        
        if (f'x{var1}', f'x{var2}') in constr_var:
            
            constr_index = constr_var[f'x{var1}', f'x{var2}'] 
            constraints[constr_index].relations += allowedTuples
            constraints[constr_index].relations = list(set(constraints[constr_index].relations))
        
        else:
            constr = Constraint(index, (f'x{var1}', f'x{var2}'), allowedTuples, domain_product)
            constraints[f'c{index}'] = constr
            constraints[f'c{index}'].relations = list(set(constraints[f'c{index}'].relations))
            
            variables[f'x{var1}'].constraints.append(f'c{index}')
            variables[f'x{var2}'].constraints.append(f'c{index}')
            
            constr_var[(f'x{var1}', f'x{var2}')] = f'c{index}'

            index += 1
    return variables, constraints

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
        domain_product = constraints[constraint].product
        allowed_tuples = len(constraints[constraint].relations)
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
    data.to(device)
    return data, var_constr_index, constr_var_index

def create_data_batch(variables_batch, constraints_batch):
    data_batch = []
    var_constr_index_batch = []
    constr_var_index_batch = []

    for variables, constraints in zip(variables_batch, constraints_batch):
        variable_info = get_variable_info(variables)
        constraint_info = get_constraint_info(variables, constraints)

        edge_index, var_constr_index, constr_var_index = generate_edge_index(variables, constraints)
        data = Data(x=None, edge_index=edge_index)
        data.x = torch.tensor(variable_info + constraint_info, dtype=torch.float)
        data.num_nodes = len(var_constr_index) + len(constr_var_index)

        data_batch.append(data)
        var_constr_index_batch.append(var_constr_index)
        constr_var_index_batch.append(constr_var_index)

    return data_batch, var_constr_index_batch, constr_var_index_batch

class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, init_input_dim, init_output_dim, num_layers):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_layers = num_layers

        self.fc_variable   = nn.Linear(input_dim, output_dim).to(device)
        self.fc_constraint = nn.Linear(input_dim, output_dim).to(device)
        
        self.fc_init_variables = nn.Linear(init_input_dim, init_output_dim).to(device)
        self.fc_init_constraints = nn.Linear(init_input_dim, init_output_dim).to(device)
        
        self.fc_q_value = nn.Linear(output_dim * 2, 1).to(device)
    
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
        zeros_tensor = torch.zeros((1, self.output_dim)).to(device)

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
    
    def predict(self, var_constr_index, constr_var_index, data, flag=0):
        # 创建变量和约束每一层的特征张量
        self.initialize_layer_zero(data.x, len(var_constr_index))
        self.aggregate(var_constr_index, constr_var_index, data.x)
        Q = self.embedding_Q()
        for var_index in range(len(var_constr_index)):
                if int(data.x[var_index][1]):
                    Q[var_index] = float('inf')
        if flag == 1:
            return torch.argmin(Q).item()
        elif flag == 2:
            return torch.min(Q)
        return Q

variables, constraints = parser_data(f'output/problems/{0}.txt')
 
data, var_constr_index, constr_var_index = create_data(variables, constraints)

target_net = GNN(input_dim=66, output_dim=32, init_input_dim = 2, init_output_dim=32, num_layers=2).to(device)
Q = target_net.predict(var_constr_index, constr_var_index, data)