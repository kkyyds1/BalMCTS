from copy import deepcopy
import ipdb
from itertools import product
#------------------------------------------------------------------------------------
# Constraint Propagation
#------------------------------------------------------------------------------------
def constraint_propagation(assignment, variables, constraints):
    updated_variables = deepcopy(variables)
    updated_constraints = deepcopy(constraints)
    queue = list(assignment.keys())
    visited = set()  # 添加一个集合来记录已经访问过的变量

    def update_domain(var):
        for name, constraint in updated_constraints.items():
            (var1, var2), valid_pairs = constraint.variables, constraint.relations
            if var1 == var:
                # updated_valid_pairs = [(val1, val2, weight) for (val1, val2, weight) in valid_pairs if val1 in variables[var1].domain]
                updated_valid_pairs = [(val1, val2) for (val1, val2) in valid_pairs if val1 in variables[var1].domain]
                if len(updated_valid_pairs) != len(valid_pairs):
                    # updated_variables[var2].domain = list(set(val2 for (val1, val2, weight) in updated_valid_pairs) & set(updated_variables[var2].domain))
                    updated_variables[var2].domain = list(set(val2 for (val1, val2) in updated_valid_pairs) & set(updated_variables[var2].domain))
                    if var2 not in queue and var2 not in visited:  # 只有当变量没有被访问过时，才把它添加到队列中
                        queue.append(var2)
            
            elif var2 == var:
                # updated_valid_pairs = [(val1, val2, weight) for (val1, val2, weight) in valid_pairs if val2 in variables[var2].domain]
                updated_valid_pairs = [(val1, val2) for (val1, val2) in valid_pairs if val2 in variables[var2].domain]
                if len(updated_valid_pairs) != len(valid_pairs):
                    # updated_variables[var1].domain = list(set(val1 for (val1, val2, weight) in updated_valid_pairs) & set(updated_variables[var1].domain))
                    updated_variables[var1].domain = list(set(val1 for (val1, val2) in updated_valid_pairs) & set(updated_variables[var1].domain))
                    
                    if var1 not in queue and var1 not in visited:  # 只有当变量没有被访问过时，才把它添加到队列中
                        queue.append(var1)
    
    def update_constraints():
        for name, constraint in updated_constraints.items():
            (var1, var2), valid_pairs = constraint.variables, constraint.relations
            var1_domain = updated_variables[var1].domain
            var2_domain = updated_variables[var2].domain
            # updated_valid_pairs = [(val1, val2, weight) for (val1, val2, weight) in valid_pairs if val1 in var1_domain and val2 in var2_domain]
            updated_valid_pairs = [(val1, val2) for (val1, val2) in valid_pairs if val1 in var1_domain and val2 in var2_domain]
            updated_constraints[name].relations = updated_valid_pairs  # 应该更新constraint的relations，而不是update_constraints的relations

    while queue:
        var = queue.pop(0)
        visited.add(var)  # 把变量添加到访问过的集合中
        update_domain(var)
    update_constraints()
   
    variables_set = []
    for k, v in updated_constraints.items():
        variables_set.append(v.variables[0])
        variables_set.append(v.variables[1])
    
    variables_invalid = set(variables) - set(variables_set)
    if variables_invalid:
        for var in variables_invalid:
            updated_variables[var].domain = []
    
    return updated_variables, updated_constraints

    

#------------------------------------------------------------------------------------
# Check Constraints
#------------------------------------------------------------------------------------
def check_constraints(assignment, variables, constraints):
    for var in variables.values():
        if var.constraints == []:
            return False
    
    for constraint in constraints.values():
        # (var1, var2), valid_pairs = constraint.variables, [(val1, val2) for (val1, val2, weight) in constraint.relations]
        (var1, var2), valid_pairs = constraint.variables, [(val1, val2) for (val1, val2) in constraint.relations]
        if var1 in assignment and var2 in assignment:
            pair = (assignment[var1], assignment[var2])
            if pair not in valid_pairs:
                return False
    return True

def constraint_tightness(variables, constraints):
    constraint_tightness = {}
    for name, constr in constraints.items():
        # 计算当前允许的元组的数量
        allowed_tuples = len([pair for pair in constr.relations if pair[0] in variables[constr.variables[0]].domain and pair[1] in variables[constr.variables[1]].domain])
        # 计算所有可能元组的数量
        possible_tuples = len(variables[constr.variables[0]].domain) * len(variables[constr.variables[1]].domain)
        # 计算约束紧致度
        constraint_tightness[name] = 1 - allowed_tuples / possible_tuples
    return constraint_tightness 