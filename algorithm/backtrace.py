from gnn.embedding import create_data, GNN
from gnn.ExperienceReplay import *
from algorithm.choose_variable import *
from algorithm.assign_value import *
from algorithm.constraint_propagation import *
from entry.data import Variable, Constraint

import torch.optim as optim

#------------------------------------------------------------------------------------
# Check Constraints
#------------------------------------------------------------------------------------
def check_constraints(assignment, constraints):
    for constraint in constraints.values():
        (var1, var2), valid_pairs = constraint.variables, constraint.relations
        if var1 in assignment and var2 in assignment:
            pair = (assignment[var1], assignment[var2])
            if pair not in valid_pairs:
                return False
    return True

# def backtrack(assignment, variables,  constraints, target_net, epsilon, value_selector, var_selector, count):
#     domains = [var.domain for var in variables.values()]
#     count[0] += 1
    
#     # 如果所有变量都已赋值，那么就找到了一个解决方案
#     if len(assignment) == len(variables):
#         # 打印搜索次数
#         print(count[0])
#         yield assignment
#         return

#     # 选择一个还没有赋值的变量
#     action = choose_variable(assignment, variables, domains, constraints, target_net, epsilon, var_selector)
#     state = create_data(variables, constraints)
    
#     # 根据你的策略选择值并进行赋值
#     for local_assignment, variables in assign_value(action, assignment, variables, domains, constraints, value_selector):
        
#         # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
#         if check_constraints(local_assignment, constraints):
#             update_variables = constraint_propagation(local_assignment, variables, domains, constraints)
            
#             next_state = create_data(variables, constraints)
            
#             # 把经验放到缓冲区
#             if len(assignment) == len(variables):
#                 epr.add((state, action, next_state, 1, 1))
#             else:
#                 epr.add(state, action, next_state, 1, 0)
            
#             yield from backtrack(local_assignment, update_variables,  constraints, target_net, epsilon, value_selector, var_selector, count)
    
epr = ExperienceReplayBuffer(1000)

def backtrack(assignment, variables, constraints, online_net, target_net, epsilon, value_selector, var_selector, train_freq, update_freq):
    stack = [(assignment, variables)]
    count = 0
    losses = []
    while stack:
        assignment, variables = stack.pop()
        domains = [var.domain for var in variables.values()]
        # 如果所有变量都已赋值，那么就找到了一个解决方案
        if len(assignment) == len(variables):
            # 打印搜索次数
            print(count)
            yield assignment, losses
            losses = []

            continue

        # 选择一个还没有赋值的变量
        action = choose_variable(assignment, variables, domains, constraints, target_net, epsilon, var_selector)
        state, var_constr_index, constr_var_index = create_data(variables, constraints)
        action_index = int(action[1:])
        # 根据你的策略选择值并进行赋值
        for local_assignment, variables in assign_value(action, assignment, variables, domains, constraints, value_selector):

            # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
            if check_constraints(local_assignment, constraints):
                count += 1

                update_variables = constraint_propagation(local_assignment, variables, domains, constraints)

                next_state, next_var_constr_index, next_constr_var_index = create_data(variables, constraints)
                # 把经验放到缓冲区
                if len(assignment) == len(variables):
                    epr.add((state, var_constr_index, constr_var_index, action_index, next_state, next_var_constr_index, next_constr_var_index, 1, 1))
                else:
                    epr.add((state, var_constr_index, constr_var_index, action_index, next_state, next_var_constr_index, next_constr_var_index, 1, 0))
                
                # nstep采样， 更新online net
                if count % train_freq == 0:
                    loss = GNN.samples(target_net, online_net, epr, batch_size=5)
                    losses.append(loss[0])

                # 把online net 的参数复制给target net
                if count % update_freq == 0:
                    target_net = GNN.update_model(target_net, online_net)
                
                stack.append((local_assignment, update_variables))
