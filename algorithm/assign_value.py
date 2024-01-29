import random
import ipdb
from copy import deepcopy
from collections import defaultdict
def assign_value_lcv(var, assignment, variables, domains, constraints):
    # 获取变量的所有可能值
    try:
        var_index = variables[var].index
    except:
        ipdb.set_trace()
    values = domains[var_index]

    # 计算每个值的约束数量
    constraints_count = {value: count_constraints(var, value, assignment, variables, constraints) for value in values}

    # 按照约束数量从小到大排序值
    sorted_values = sorted(values, key=lambda value: constraints_count[value])
    return sorted_values

def count_constraints(var, value, assignment, variables, constraints):
    # 计算给定值会对其他变量产生多少约束
    count = 0
    for constraint in constraints.values():
        # valid_pairs = [(val1, val2) for val1, val2, weight in constraint.relations]
        valid_pairs = [(val1, val2) for val1, val2 in constraint.relations]
        if var in constraint.variables:
            other_var = constraint.variables[0] if constraint.variables[1] == var else constraint.variables[1]
            if other_var not in assignment:
                for other_value in variables[other_var].domain:
                    if (value, other_value) not in valid_pairs and (other_value, value) not in valid_pairs:
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

def assign_value_min(var, assignment, variables, domains, constraints):
    # 获取变量的所有可能值
    var_index = variables[var].index
    values = domains[var_index]
    values = sorted(values)
    return values

def assign_value_argmin(var, assignment, variables, domains, constraints):
    temp = defaultdict(list)
    temp_dict = {}
    for constr in constraints.values():
        if var == constr.variables[0]:
            min_value = [val3 for val1, val2, val3 in constr.relations]
            for val in list(set([val1 for val1, val2, val3 in constr.relations if val3 == min(min_value)])):
                if val not in temp:
                    temp[val] = 0
                temp[val] += 1
    temp = dict(sorted(temp.items(), key=lambda x : x[1], reverse=True))
    return tuple(temp.keys())

def assign_value(var, assignment, variables, domains, constraints, value_selector):
    domains_copy = deepcopy(domains)
    variables_copy = deepcopy(variables)
    # 根据策略选择值
    if value_selector == 'LCV':
        values = assign_value_lcv(var, assignment, variables_copy, domains_copy, constraints)
    elif value_selector == 'RANDOM':
        values = assign_value_random(var, assignment, variables_copy, domains_copy, constraints)
    elif value_selector == 'MIN':
        values = assign_value_min(var, assignment, variables_copy, domains_copy, constraints)
    elif value_selector == 'argmin':
        values = assign_value_argmin(var, assignment, variables_copy, domains_copy, constraints)
    for value in values:
        # 对变量进行赋值
        yield value