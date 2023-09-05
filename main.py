from gnn.embedding import create_data, GNN
from algorithm.choose_variable import choose_ddeg, choose_first_unbound, choose_min_q, choose_mindom
from algorithm.assign_value import assign_value
from algorithm.backtrace import backtrack
from entry.parser import parser_data
from entry.RBGeneration import generate_instance_files
import matplotlib.pyplot as plt

import ipdb
# generate_instance_files(2, 15, 0.7, 3, 0.21, "output/problems/", 10)

variables, constraints = parser_data('output/problems/0.txt')
data = create_data(variables, constraints)

old_net = GNN(input_dim=22, output_dim=10, init_input_dim = 2, init_output_dim=10, num_layers=3)
new_net = GNN(input_dim=22, output_dim=10, init_input_dim = 2, init_output_dim=10, num_layers=3)

# action = old_net.predict(variables, constraints, data)
init_assign = {}
solutions = []
losses = []
for _ in range(5):
    for solution, loss in backtrack(init_assign, variables, constraints, new_net, old_net, 0.2, 'LCV', 'CHOOSE_MIN_Q', 50, 500):
        solutions.append(solution)
        losses.extend(loss)
with open(f'loss.txt', 'w') as f:
    for loss_value in losses:
        f.write(str(loss_value) + '\n')



