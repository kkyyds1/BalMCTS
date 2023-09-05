from copy import deepcopy
import ipdb
#------------------------------------------------------------------------------------
# Constraint Propagation
#------------------------------------------------------------------------------------
def constraint_propagation(assignment, variables, domains, constraints):
    updated_domains = deepcopy(domains)
    updated_variables = deepcopy(variables)
    for constraint in constraints.values():
        (var1, var2), valid_pairs = constraint.variables, constraint.relations
        if var1 in assignment:
            updated_variables[var2].domain = updated_domains[variables[var2].index]
        elif var2 in assignment:
            updated_variables[var1].domain = updated_domains[variables[var1].index]
    return updated_variables


# updated_constraints[f'c{constraint.index}'].relations 