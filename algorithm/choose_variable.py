from gnn.embedding import create_data
from copy import deepcopy
import torch
import math
import numpy as np
import ipdb
def choose_variable(assignment, variables, domains, constraints, ddqn=None, epsilon=0.2, node=None, var_selector='CHOOSE_MIN_Q', phase='Training'):
    if var_selector == 'CHOOSE_FIRST_UNBOUND':
        res = choose_first_unbound(assignment, variables)
    elif var_selector == 'CHOOSE_MIN_Q':
        res = choose_min_q(assignment, variables, domains, constraints, ddqn.target_net, epsilon, phase)
    elif var_selector == 'CHOOSE_DDEG':
        res = choose_ddeg(assignment, variables, domains, constraints)
    elif var_selector == 'CHOOSE_RANDOM':
        res = choose_random(assignment, variables, constraints)
    elif var_selector == 'CHOOSE_MINDOM':
        res = choose_mindom(assignment, variables, domains)
    elif var_selector == 'CHOOSE_MCTS':
        res = choose_mcts(assignment, variables, domains, constraints, node)
    return res

def choose_first_unbound(assignment, variables):
    # 选择一个还没有赋值的变量
    unassigned = [v for v in variables if v not in assignment] 
    return unassigned

def choose_min_q(assignment, variables, domains, constraints, target_net, epsilon, phase):
    data, var_constr_index, constr_var_index = create_data(variables, constraints)
    if phase == 'Training' and np.random.rand() < epsilon:
        return choose_ddeg(assignment, variables, domains, constraints), None
    Q = target_net.predict(var_constr_index, constr_var_index, data)
    
    index = torch.argmin(Q).item()
    _, indices = torch.sort(Q, dim=0)
    indices = indices.squeeze().tolist()
    for var_index in range(len(var_constr_index)):
        if int(data.x[var_index][1]):
            indices.remove(var_index)
    return indices, Q

def choose_ddeg(assignment, variables, domains, constraints):
    # 计算每个未赋值变量的动态紧致度
    unassigned = [v for v in variables if v not in assignment]
    num_unbounded = {}
    for name, constr in constraints.items():
        tmp_count = 0
        for var in constr.variables:
            if var not in unassigned:
                tmp_count += 1
        num_unbounded[name] = tmp_count
    
    ddeg = {}
    for name, var in variables.items():
        if name not in unassigned:
            continue
        tmp_count = 1
        for constr in var.constraints:
            if num_unbounded[constr] >= 1:
                tmp_count += 1
        ddeg[name] = tmp_count

    dynamic_tightness = {}
    for name, var in variables.items():
        if name in unassigned:
            index = int(name[1:])
            dynamic_tightness[index] = len(var.domain) / ddeg[name]
    dynamic_tightness = sorted(dynamic_tightness, key=dynamic_tightness.get)
    # 选择动态紧致度最高的变量
    return dynamic_tightness

def choose_mcts(assignment, variables, domains, constraints, node):
    # 计算每个未赋值变量的动态紧致度
    pi = {}
    unassigned = [v for v in variables if v not in assignment]
    data, var_constr_index, constr_var_index = create_data(variables, constraints)
    Q = target_net.predict(var_constr_index, constr_var_index, data)
    index = torch.argmin(Q).item()
    _, indices = torch.sort(Q, dim=0)
    indices = indices.squeeze().tolist()
    for var_index in range(len(var_constr_index)):
        pi[var_index] = Q[var_index] - 0.5 * math.sqrt(math.log(node.visits + 1) / node.children[var_index].visits) - node.alpha
        if int(data.x[var_index][1]):
            indices.remove(var_index)
    pi = sorted(pi, key=pi.get, reverse=True)
    return pi[0]

def choose_mindom(assignment, variables, domains):
    # 计算每个未赋值变量的域的大小
    unassigned = [v for v in variables if v not in assignment]
    domain_sizes = {var: len(domains[variables.index(var)]) for var in unassigned}
    domain_sizes = sorted(domain_sizes, key = domain_sizes.get, reverse=True)
    # 选择域大小最小的变量
    return min(domain_sizes, key=domain_sizes.get)

def compute_ddeg(var, variables, constraints):
    rel_constraints = 0
    for constr in constraints.values():
        if var in constr.variables:
            rel_constraints += len(constr.relations)
    return rel_constraints

def choose_random(assignment, variables, constraints):
    # 选择一个随机的未赋值变量
    unassigned = [v for v in variables if v not in assignment]
    return unassigned

def compute_table_constr_tightness(variables, constraints):
    # Compute the product of domain sizes
    tightness = {}
    domain_product = 1
    for constr in constraints:
        domain_product = constraints[constr].product

        num_active_tuples = len(constraints[constr].relations)
        tightness[constr] = (2, 1 - num_active_tuples / domain_product)
    return tightness

    # related_constraints = [constr for constr in constraints.values() if var in constr.variables]

    # for constr in related_constraints:
    #     other_var = constr.variables[0] if constr.variables[1] == var else constr.variables[1]
    #     constr.relations = [rel for rel in constr.relations if (constr.variables[0] == var and rel[0] == assignment[var]) or (constr.variables[1] == var and rel[1] == assignment[var])]
    #     updated_constraints[f'c{constr.index}'].relations = constr.relations
    #     updated_variables[other_var].domain = [rel[1] if constr.variables[0] == var else rel[0] for rel in constr.relations]

    # return updated_constraints, updated_variables