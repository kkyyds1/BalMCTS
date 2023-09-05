from gnn.embedding import create_data, GNN
from algorithm.choose_variable import choose_ddeg, choose_first_unbound, choose_min_q, choose_mindom
from algorithm.assign_value import assign_value
from algorithm.backtrace import backtrack
from entry.parser import parser_data
from entry.RBGeneration import generate_instance_files
import matplotlib.pyplot as plt

import ipdb

k = 2  # Arity
n = 15  # nVar
alpha = 0.7  # domain scale
r = 3  # nConstr
p = 0.21  # tightness
num_instance = 100
model_path = 'output/models/model_0'
phase = 'Evaluation'

generate_instance_files(k, n, alpha, r, p, "output/problems/", num_instance)

old_net = GNN(input_dim=34, output_dim=16, init_input_dim = 2, init_output_dim=16, num_layers=3)
new_net = GNN(input_dim=34, output_dim=16, init_input_dim = 2, init_output_dim=16, num_layers=3)

if phase == 'Evaluation':
    GNN.load_model(old_net, model_path)
    GNN.load_model(new_net, model_path)

for i in range(num_instance):
    variables, constraints = parser_data(f'output/problems/{i}.txt')
    init_assign = {}
    solutions = []
    losses = []
    
    for _ in range(100):
        for solution, loss in backtrack(init_assign, variables, constraints, new_net, old_net, 0.2, 'LCV', 'CHOOSE_DDEG', 50, 500, 'Evaluation'):
            solutions.append(solution)
            losses.extend(loss)
    
    with open(f'output/losses/loss_{i}.txt', 'w') as f:
        for loss_value in losses:
            f.write(str(loss_value) + '\n')

    GNN.save_model(old_net, f'output/models/model_{i * 100}')
    plt.plot(losses[::10])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.savefig(f'output/pictures/loss_plot_{i}.png')

