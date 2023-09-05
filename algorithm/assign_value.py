import random
import ipdb
from copy import deepcopy
def assign_value_lcv(var, assignment, variables, domains, constraints):
    # 获取变量的所有可能值
    var_index = variables[var].index
    values = domains[var_index]

    # 计算每个值的约束数量
    constraints_count = {value: count_constraints(var, value, assignment, constraints) for value in values}

    # 按照约束数量从小到大排序值
    sorted_values = sorted(values, key=lambda value: constraints_count[value])

    return sorted_values

def count_constraints(var, value, assignment, constraints):
    # 计算给定值会对其他变量产生多少约束
    count = 0
    for constraint in constraints.values():
        (var1, var2), valid_pairs = constraint.variables, constraint.relations
        if var1 == var and var2 in assignment:
            pair = (value, assignment[var2])
            if pair not in valid_pairs:
                count += 1
        elif var2 == var and var1 in assignment:
            pair = (assignment[var1], value)
            if pair not in valid_pairs:
                count += 1
    return count

def assign_value_random(var, assignment, variables, domains, constraints):
    # 获取变量的所有可能值
    var_index = variables[var].index
    values = domains[var_index]
    random_values = []
    for _ in range(len(values)):
        # 随机选择一个值
        value = random.choice(values)
        values.remove(value)
        random_values.append(value)
    return random_values

def assign_value(var, assignment, variables, domains, constraints, value_selector):
    domains_copy = deepcopy(domains)
    variables_copy = deepcopy(variables)
    # 根据策略选择值
    if value_selector == 'LCV':
        values = assign_value_lcv(var, assignment, variables_copy, domains_copy, constraints)
    elif value_selector == 'RANDOM':
        values = assign_value_random(var, assignment, variables_copy, domains_copy, constraints)
    
    for value in values:
        # 对变量进行赋值
        local_assignment = assignment.copy()
        local_assignment[var] = value
        variables_copy[var].is_assigned = 1
        variables_copy[var].domain = [value]
        yield local_assignment, variables_copy