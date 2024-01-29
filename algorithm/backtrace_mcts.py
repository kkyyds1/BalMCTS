from gnn.embedding import create_data, GNN
from gnn.ExperienceReplay import *
from algorithm.choose_variable import *
from algorithm.assign_value import *
from algorithm.constraint_propagation import *
from entry.data import Variable, Constraint
import torch.optim as optim

def backtrack_training(assignment, solutions, variables, constraints, net, value_selector='LCV', var_selector='CHOOSE_MCTS', train_freq=16, states = [], actions = [], values = [], count = [0], max_steps=100):
    domains = [var.domain for var in variables.values()]
    
    # 如果所有变量都已赋值，那么就找到了一个解决方案
    if len(assignment) == len(variables): 
        return 1
    
    if count[0]  > max_steps:
        return max_steps

    # 选择一个还没有赋值的变量
    actions = choose_variable(assignment, variables, domains, constraints, var_selector= var_selector)
    state = create_data(variables, constraints)
    Q = []
    if tuple(assignment.keys()) in solutions:
        return solutions[tuple(assignment.keys())]
    for action in actions:
    # 根据你的策略选择值并进行赋值
        search_node = 0
        for value in assign_value(action, assignment, variables, domains, constraints, value_selector):
            count[0] += 1
            local_assignment = assignment.copy()
            local_variables = deepcopy(variables)
            local_assignment[action] = value
            local_variables[action].is_assigned = 1
            local_variables[action].domain = [value]
            update_variables, update_constraints = constraint_propagation(local_assignment, local_variables, domains, constraints)
            search_node += 1
            # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
            if check_constraints(local_assignment,update_variables, update_constraints):
                q = backtrack_training(local_assignment, solutions, update_variables, update_constraints,  ddqn = ddqn, var_selector=var_selector, train_freq=train_freq, states=states, actions=actions, values=values,count=count, max_steps=max_steps)
                Q.append(q + search_node)
                if q != max_steps:
                    break
            else:
                Q.append(max_steps)
    reward = min(Q) if Q != [] else max_steps
    solutions[tuple(assignment.keys())] = reward
    states.append(state)
    values.append(reward)
    actions.append(int(action[1:]))
    if len(actions) >= train_freq:
        net.training(states, actions, values)
        states.clear(), actions.clear(), values.clear()
    return reward
      
def backtrack_evaluation(assignment, variables, constraints, ddqn = None, value_selector='LCV', var_selector='CHOOSE_MIN_Q', count=[0], max_steps=10000):
    domains = [var.domain for var in variables.values()]
    # 如果所有变量都已赋值，那么就找到了一个解决方案
    if len(assignment) == len(variables):
        return (True, count[0])  # 如果找到解，返回搜索次数

    # 如果超过最大搜索步数，返回无解
    if count[0] >= max_steps:
        return (False, count[0])

    # 选择一个还没有赋值的变量
    actions, predict_qs = choose_variable(assignment, variables, domains, constraints, ddqn=ddqn, var_selector= var_selector, phase='Evaluation')
    action = f'x{actions[0]}'
    # 根据你的策略选择值并进行赋值
    for value in assign_value(action, assignment, variables, domains, constraints, value_selector):
        count[0] += 1
        local_assignment = assignment.copy()
        local_variables = deepcopy(variables)
        local_assignment[action] = value
        local_variables[action].is_assigned = 1
        local_variables[action].domain = [value]
        update_variables, update_constraints = constraint_propagation(local_assignment, local_variables, domains, constraints)
        
        # 如果这个赋值满足所有的约束，那么就在这个赋值的基础上递归地进行回溯搜索
        if check_constraints(local_assignment,update_variables, update_constraints):
            flag, res = backtrack_evaluation(local_assignment, update_variables, update_constraints, ddqn = ddqn, var_selector=var_selector, count=count, max_steps=max_steps)
            if flag:  # 如果找到解，返回搜索次数
                return (True, res)
        else:
            continue  # 如果不满足约束，继续搜索
    return (False, count[0])  # 如果所有值都试过了，还没有找到解，返回无解
