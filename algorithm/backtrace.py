from gnn.embedding import create_data, GNN
from gnn.ExperienceReplay import *
from algorithm.choose_variable import *
from algorithm.assign_value import *
from algorithm.constraint_propagation import *
from entry.data import Variable, Constraint
import torch.optim as optim

# def backtrack(assignment, variables, constraints, ddqn=None, epsilon=0.2, value_selector='LCV', var_selector='CHOOSE_DDEG', train_freq=16, update_freq=128, phase=''):
#     counts = []
#     count = 0
#     stack = [(assignment, variables, constraints, count)]
#     variables_batch = []
#     constraints_batch = []
#     while stack:
#         new_assignment, new_variables, new_constraints, new_count = stack.pop()
#         new_domains = [var.domain for var in new_variables.values()]
        
#         # 如果所有变量都已赋值，那么就找到了一个解决方案
#         if len(new_assignment) == len(new_variables):
#             yield new_assignment
#             continue
        
#         # 选择一个还没有赋值的变量
#         action = choose_variable(new_assignment, new_variables, new_domains, new_constraints, ddqn=ddqn, epsilon=epsilon, var_selector = var_selector, phase=phase)
#         new_count += 1
#         # 根据你的策略选择值并进行赋值
#         for value in assign_value(action, new_assignment, new_variables, new_domains, new_constraints, value_selector):
            
#             local_assignment = new_assignment.copy()
#             local_variables = deepcopy(new_variables)
#             local_assignment[action] = value
#             local_variables[action].is_assigned = 1
#             local_variables[action].domain = [value]
#             update_variables, update_constraints = constraint_propagation(local_assignment, local_variables, new_domains, new_constraints)
            
#             # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
#             if check_constraints(local_assignment,update_variables, update_constraints):
#                 if phase == 'Training':
#                 #     ddqn.store_experience(assignment, variables, constraints, local_assignment, update_variables, update_constraints, action, count, train_freq, update_freq)
#                     variables_batch.append(update_variables)
#                     constraints_batch.append(update_constraints)
#                     counts.append(new_count)
#                 stack.append((local_assignment, update_variables, update_constraints, new_count))
#             else:
#                 variables_batch.append(update_variables)
#                 constraints_batch.append(update_constraints)
#                 counts.append(1000)
#             # if len(variables_batch) == 16:
                
#             #     ddqn.training(variables_batch, constraints_batch)
#             #     constraints_batch = []
#             #     variables_batch = [] 
           
def backtrack(assignment, variables, constraints, ddqn=None, epsilon=0.2, value_selector='LCV', var_selector='CHOOSE_DDEG', train_freq=16, update_freq=128, phase='', states = [], actions = [], values = [], count = [0]):
    # 如果所有变量都已赋值，那么就找到了一个解决方案
    if len(assignment) == len(variables):
        return True, 1

    if count[0] > 10000:
        return False, count[0]
    # 选择一个还没有赋值的变量
    update_variables, update_constraints = constraint_propagation(assignment, variables, constraints)
    domains = [var.domain for var in variables.values()]
    actions = choose_variable(assignment, update_variables, domains, update_constraints, ddqn=ddqn, epsilon=epsilon, var_selector = var_selector, phase=phase)
    action = f'x{actions[0]}'
    state = create_data(update_variables, update_constraints)
    q = 0
    Q = []
    step = 0
    # 根据你的策略选择值并进行赋值
    for value in assign_value(action, assignment, update_variables, domains, update_constraints, value_selector):
        step += 1
        count[0] += 1
        local_assignment = assignment.copy()
        local_assignment[action] = value
        update_variables[action].is_assigned = 1
        update_variables[action].domain = [value]
        # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
        if check_constraints(local_assignment, update_variables, update_constraints):
            flag, res = backtrack(local_assignment, update_variables, update_constraints, ddqn=ddqn, epsilon=epsilon, value_selector=value_selector, var_selector=var_selector, train_freq=train_freq, update_freq=update_freq, phase=phase, states=states, actions=actions, values=values,count=count)
            if flag:
                return True, step + res
    return False, step
        
        
    