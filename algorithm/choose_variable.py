from gnn.embedding import create_data
from copy import deepcopy
import torch
import numpy as np
import ipdb

def choose_variable(assignment, variables, domains, constraints, target_net, epsilon, var_selector):
    if var_selector == 'CHOOSE_FIRST_UNBOUND':
        res = choose_first_unbound(assignment, variables)
    elif var_selector == 'CHOOSE_MIN_Q':
        res = choose_min_q(assignment, variables, constraints, target_net, epsilon)
    elif var_selector == 'CHOOSE_DDEG':
        res = choose_ddeg(assignment, variables, domains, constraints)
    elif var_selector == 'CHOOSE_RANDOM':
        res = choose_random(assignment, variables, constraints)
    elif var_selector == 'CHOOSE_MINDOM':
        res = choose_mindom(assignment, variables, domains)
    return res

def choose_first_unbound(assignment, variables):
    # 选择一个还没有赋值的变量
    unassigned = [v for v in variables if v not in assignment]
    return unassigned[0]

def choose_min_q(assignment, variables, constraints, target_net, epsilon):
    data, var_constr_index, constr_var_index = create_data(variables, constraints)
    if np.random.rand() < epsilon:
        return choose_random(assignment, variables, constraints)
    else:
        Q = target_net.predict(var_constr_index, constr_var_index, data)
        for var_index in range(len(var_constr_index)):
            if int(data.x[var_index][1]):
                Q[var_index] = float('inf')
        index = torch.argmin(Q).item()
        return f'x{index}'

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
            dynamic_tightness[name] = len(var.domain) / ddeg[name]
    
    if dynamic_tightness == {}:
        return choose_first_unbound(assignment, variables)
    
    # 选择动态紧致度最高的变量
    return min(dynamic_tightness, key=dynamic_tightness.get)

def choose_mindom(assignment, variables, domains):
    # 计算每个未赋值变量的域的大小
    unassigned = [v for v in variables if v not in assignment]
    domain_sizes = {var: len(domains[variables.index(var)]) for var in unassigned}

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
    return np.random.choice(unassigned)

def compute_table_constr_tightness(variables, constraints):
    # Compute the product of domain sizes
    tightness = {}
    domain_product = 1
    for constr in constraints:
        for var in constraints[constr].variables:
            domain_product *= len(variables[var].domain)

        num_active_tuples = len(constraints[constr].relations)
        tightness[constr] = (2, 1 - num_active_tuples / domain_product)
    return tightness

    related_constraints = [constr for constr in constraints.values() if var in constr.variables]

    for constr in related_constraints:
        other_var = constr.variables[0] if constr.variables[1] == var else constr.variables[1]
        constr.relations = [rel for rel in constr.relations if (constr.variables[0] == var and rel[0] == assignment[var]) or (constr.variables[1] == var and rel[1] == assignment[var])]
        updated_constraints[f'c{constr.index}'].relations = constr.relations
        updated_variables[other_var].domain = [rel[1] if constr.variables[0] == var else rel[0] for rel in constr.relations]

    return updated_constraints, updated_variables