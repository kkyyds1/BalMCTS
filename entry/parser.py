import sys
sys.path.append('..')
import ipdb
from ast import literal_eval
from entry.data import Variable, Constraint
from collections import defaultdict
def parser_data(filepath):
    variables = {}
    constraints = {}
    relations = defaultdict(dict)
    caseFile=open(filepath,"r")
    lines=caseFile.readlines()

    # Read head
    paramValues=lines[1].split('\t')
    n = int(paramValues[0])
    d = int(paramValues[1])
    m = int(paramValues[2])
    k = int(paramValues[3])
    nb = int(paramValues[4])

    for i in range(n):
        var = Variable(i, list(range(d)), 0, [])
        variables[f'x{i}'] = var
    
    constr_var = {}
    index = 0
    for i in range(m):
        constrStrings=lines[i+3].split('|')
        
        var1, var2 = literal_eval(constrStrings[0])
        
        allowedTuples = literal_eval(constrStrings[1])
        
        domain_product = len(variables[f'x{var1}'].domain) * len(variables[f'x{var2}'].domain)
        
        if (f'x{var1}', f'x{var2}') in constr_var:
            
            constr_index = constr_var[f'x{var1}', f'x{var2}'] 
            constraints[constr_index].relations += allowedTuples
            constraints[constr_index].relations = list(set(constraints[constr_index].relations))
            
        else:
            constr = Constraint(index, (f'x{var1}', f'x{var2}'), allowedTuples, domain_product)
            constraints[f'c{index}'] = constr
            constraints[f'c{index}'].relations = list(set(constraints[f'c{index}'].relations))
            
            variables[f'x{var1}'].constraints.append(f'c{index}')
            variables[f'x{var2}'].constraints.append(f'c{index}')
            
            constr_var[(f'x{var1}', f'x{var2}')] = f'c{index}'

            index += 1
    
    # for constraint in constraints.values():
    #     variable = constraint.variables
    #     relation = constraint.relations
    #     relations[variable] = dict([((val1, val2), val3) for val1, val2, val3 in relation])
    return variables, constraints