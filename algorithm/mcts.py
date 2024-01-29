from algorithm.constraint_propagation import *
from algorithm.choose_variable import *
from algorithm.assign_value import *
from algorithm.backtrace import backtrack
from copy import deepcopy
import random
class COP:
    def __init__(self, variables, domains, constraints, weights):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.weights = weights

    def get_weight(self, assignment):
        weight = 0
        for (var1, var2), (val1, val2, w) in self.constraints.items():
            if assignment.get(var1) == val1 and assignment.get(var2) == val2:
                weight += w
        return weight

class VarNode:
    def __init__(self, var, domain, parent):
        self.var = var
        self.parent = parent
        self.children = {}
        self.domain = domain
        self.visits = 0
        self.wins = 0

class ValNode:
    def __init__(self, parent):
        self.weight = 10000
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0

class BalMCTS:
    def __init__(self, cop):
        self.cop = cop
        self.root = ValNode(0, None, self.cop.variables)
    
    def selection(self, node, assignment):
        return choose_variable(assignment, variables, domains, constraints, node, var_selector='CHOOSE_MCTS')
        
    def expansion(self, varNode, assignment):
        unassignment = set(self.cop.variables) - set(assignment.keys())
        value = random.choice(varNode.domain)
        valNode = ValNode(value, varNode)
        for var in unassignment:
            child = VarNode(var, self.cop.domains[var], valNode)
            valNode.children[var] = child
    
    def simulation(self, valNode, assignment):
        unassignment = set(self.cop.variables) - set(assignment)
        assignment_copy = deepcopy(assignment)
        for var in unassignment:
            value = random.choice(self.cop.domains[var])
            assignment_copy[var] = value
        
        return check_constraints(assignment_copy, self.cop.constraints), assignment_copy
    
    def mirror(self, assignment):
        assignment_copy = list(deepcopy(assignment).keys())
        assignment_copy[-1], assignment_copy[-2] = assignment_copy[-2], assignment_copy[-1]
        node = self.root
        for var in assignment_copy:
            node = node.children[var]
            if var not in node.children:
                node.children[var] = VarNode(var, self.cop.domains[var], node)
                node = node.children[assignment[var]]
                return
            node = node.children[assignment[var]]
                
    def backup(self, node, result, weight, assignment):
        node.visits += 1
        node.wins += result
        if type(node) == VarNode:
            self.weight = min(self.children.values(), key=lambda child: child.weight)
            
        if result == 1 and type(node) == ValNode:
            if self.copy.get_weight(assignment) < self.weight:
                node.weight = self.cop.get_weight(assignment) 
                assignment[node.parent.var] = node.value
                
        if node.parent:
            self.backup(node.parent, result, node.weight, assignment)         

    def search(self):
        assignment = {}
        node = self.root
        while node.children:
            var = self.selection(node, assignment)
            node = node.children[var]
            if var not in assignment:
                assignment[var] = node.value
        node = self.expansion(node, assignment)
        for _ in range(10):
            result, simulation_assignment = self.simulation(node, assignment)
            simulation_assignment = dict(node.assignment.items() - simulation_assignment.items())
            if result:
                self.mirror(assignment)
            self.backup(node, result, node.weight, simulation_assignment)
                
    
    
    
   
    
    
    