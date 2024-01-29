from gnn.embedding import create_data, GNN
from gnn.ExperienceReplay import *
from algorithm.choose_variable import *
from algorithm.assign_value import *
from algorithm.constraint_propagation import *
from entry.data import Variable, Constraint
import torch.optim as optim

def backtrack_training(assignment, variables, constraints, ddqn, epsilon=0.2, value_selector='LCV', var_selector='CHOOSE_MIN_Q', train_freq=16, update_freq=128, phase='Training', count = [0], step = [0], max_steps = 1000):
    # 如果所有变量都已赋值，那么就找到了一个解决方案
    if len(assignment) == len(variables) or step[0] >= max_steps:
        # ddqn.store_experience(predict_q, action, local_assignment, update_variables, update_constraints, count[0], train_freq, update_freq, T = 1)
        yield 1
        return

    # 选择一个还没有赋值的变量
    update_variables, update_constraints = constraint_propagation(assignment, variables, constraints)
    domains = [var.domain for var in update_variables.values()]
    actions, predict_qs = choose_variable(assignment, update_variables, domains,  update_constraints, ddqn = ddqn, var_selector= var_selector, phase=phase)
    action = f'x{actions[0]}'
    if predict_qs is not None:
        predict_q = predict_qs[actions[0]]
    
    # 根据你的策略选择值并进行赋值
    for value in assign_value(action, assignment, update_variables, domains, update_constraints, value_selector):
        step[0] += 1
        local_assignment = assignment.copy()
        local_variables = deepcopy(update_variables)
        local_assignment[action] = value
        local_variables[action].is_assigned = 1
        local_variables[action].domain = [value]
        # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
        if check_constraints(local_assignment,local_variables, update_constraints):
            
            for res in backtrack_training(local_assignment, local_variables, update_constraints, ddqn=ddqn, epsilon=epsilon, value_selector=value_selector, var_selector=var_selector, train_freq=train_freq, update_freq=update_freq, phase=phase, count = count, step = step, max_steps = max_steps):
                if res and predict_qs is not None:
                    count[0] += 1
                    ddqn.store_experience(predict_q, action, local_assignment, local_variables, update_constraints, count[0], train_freq, update_freq, T = res + 1)
                    yield res + 1
                else:
                    continue
    if predict_qs is not None:
        count[0] += 1
        ddqn.store_experience(predict_q, action, assignment, update_variables, update_constraints, count[0], train_freq, update_freq, T = 1)
    yield 1

def backtrack_evaluation(assignment, variables, constraints, ddqn, epsilon=0.2, value_selector='LCV', var_selector='CHOOSE_MIN_Q', phase='Evaluation', count = [0], max_steps = 10000):
    domains = [var.domain for var in variables.values()]
    
    # 如果所有变量都已赋值，那么就找到了一个解决方案
    if len(assignment) == len(variables):
        return True, count[0]

    if count[0] >= max_steps:
        return False, count[0]
    
    # 选择一个还没有赋值的变量
    try:
        actions, predict_qs = choose_variable(assignment, variables, domains,  constraints, ddqn = ddqn, var_selector= var_selector, phase=phase)
    except:
        ipdb.set_trace()
    action = f'x{actions[0]}'
    # 根据你的策略选择值并进行赋值
    for value in assign_value(action, assignment, variables, domains, constraints, value_selector):
        count[0] += 1
        local_assignment = assignment.copy()
        local_variables = deepcopy(variables)
        local_assignment[action] = value
        local_variables[action].is_assigned = 1
        local_variables[action].domain = [value]
        update_variables, update_constraints = constraint_propagation(local_assignment, local_variables, constraints)
        # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
        if check_constraints(local_assignment,update_variables, update_constraints):
            res = backtrack_evaluation(local_assignment, local_variables, update_constraints, ddqn=ddqn, epsilon=epsilon, value_selector=value_selector, var_selector=var_selector,count = count, max_steps = max_steps)
            
            if res[0]:
                return res
    return False, max_steps