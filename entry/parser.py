import sys
sys.path.append('..')
import ipdb
from ast import literal_eval
from entry.data import Variable, Constraint
def parser_data(filepath):
    variables = {}
    constraints = {}

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
    
    for i in range(m):
        constrStrings=lines[i+3].split('|')
        var1, var2 = literal_eval(constrStrings[0])
        allowedTuples = literal_eval(constrStrings[1])

        constr = Constraint(i, (f'x{var1}', f'x{var2}'), allowedTuples)
        constraints[f'c{i}'] = constr

        variables[f'x{var1}'].constraints.append(f'c{i}')
        variables[f'x{var2}'].constraints.append(f'c{i}')
    return variables, constraints