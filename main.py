from gnn.embedding import create_data, GNN, DDQN
from algorithm.choose_variable import choose_ddeg, choose_first_unbound, choose_min_q, choose_mindom
from algorithm.assign_value import assign_value
from algorithm import backtrace, constraint_propagation
from algorithm import backtrace_ddqn
from algorithm import backtrace_simple
from algorithm.mcts import Node
from entry.parameter import *
from entry.parser import parser_data
from entry.RBGeneration import generate_instance_files
import matplotlib.pyplot as plt
import math
import ipdb
k = 2  # Arity
n = 40 # nVar
alpha = 0.7   # domain scale
r =2 # nConstr
p = 0.21  # tightness
num_instance = 100
model_path = '/home/xiaoyingkai/learningcsp/output/models/ddqn__2_15_298_2'
phase = 'Training'
iteration = 1000
generate_instance_files(k, n, alpha, r, p, "output/problems/", num_instance, if_sat=True)
ddqn = DDQN(input_dim=66, output_dim=32, init_input_dim = 2, init_output_dim=32, num_layers=2, phase=phase, model_path=model_path)

def training_simple():
    states, actions, values=[], [], []
    for epoch in range(iteration):
        print('iteration:', epoch)
        for i in range(num_instance):
            print('num_instance:', i)
            variables, constraints = parser_data(f'output/problems/{i}.txt')
            init_assign = {}
            solutions = {}
            count = [0]
            sol_num = 0
            for sol in backtrace_simple.backtrack_training(init_assign, solutions, variables, constraints, ddqn, 'LCV', 'CHOOSE_DDEG', 4096, states=states, actions=actions, values=values, count=count):
                print(sol)
                sol_num += 1
                if sol_num > 20:
                    break
                pass
        if epoch > 0 and epoch % 10 == 0:
            ddqn.save_model(f'./output/models/simple_model__{k}_{n}_{epoch}_{i}')

def evaluation_simple():
    win  = 0
    lose = 0
    model_path_ddqn = '/home/xiaoyingkai/learningcsp/output/models/ddqn__2_15_500_2'
    model_path_simple = '/home/xiaoyingkai/learningcsp/output/models/simple_model__2_25_70_99'
    ddqn = DDQN(input_dim=66, output_dim=32, init_input_dim = 2, init_output_dim=32, num_layers=2, phase=phase, model_path=model_path_ddqn)
    simple = DDQN(input_dim=66, output_dim=32, init_input_dim = 2, init_output_dim=32, num_layers=2, phase=phase, model_path=model_path_simple)
    res_1 = 0
    res_2 = 0
    res_3 = 0
    for i in range(num_instance):
        variables, constraints = parser_data(f'output/problems/{i}.txt')
        
        variables, constraints = constraint_propagation.constraint_propagation({}, variables, constraints)

        flag_1, count_1 = backtrace_ddqn.backtrack_evaluation({}, variables, constraints, ddqn = ddqn,  value_selector='LCV', var_selector='CHOOSE_MIN_Q', count=[0])
        
        flag_2, count_2 =  backtrace_simple.backtrack_evaluation({}, variables, constraints, ddqn = simple,  value_selector='LCV', var_selector='CHOOSE_MIN_Q', count=[0])
        
        # flag_3, count_3 = backtrace.backtrack({}, variables, constraints, ddqn=None, epsilon=0.2, value_selector='LCV', var_selector='CHOOSE_DDEG')
        # res_1 += count_1
        # res_2 += count_2
        # res_3 += count_3
        print( flag_2, count_2, flag_1, count_1)
        # print(flag_1, count_1, flag_2, count_2, flag_3, count_3)
        # if count_3 > count_2 or (not flag_3 and flag_2):
        #     win += 1
        
        # elif count_3 < count_2 or (not flag_2 and flag_3):
        #     lose += 1
    
    # print('win', win, 'lose', lose)
    # print(res_3 / num_instance)
    # print(res_2 // num_instance)

def training_ddqn():
    count = [0]
    for epoch in range(299, iteration):
        print('iteration:', epoch)
        for i in range(num_instance):
            print('num_instance:', i)
            variables, constraints = parser_data(f'output/problems/{i}.txt')
            init_assign = {}
            for sol in backtrace_ddqn.backtrack_training(init_assign, variables, constraints, ddqn=ddqn, epsilon=0.2, value_selector='LCV', var_selector='CHOOSE_MIN_Q', train_freq=4096, update_freq=9192, phase=phase, count=count, step=[0]):
                pass

        ddqn.save_model(f'./output/models/ddqn__{k}_{n}_{epoch}_{2}')

# evaluation_simple()  
# import torch
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 
# variables, constraints = parser_data(f'output/problems/{0}.txt')
# data, var_constr_index, constr_var_index = create_data(variables, constraints)

# target_net = GNN(input_dim=67, output_dim=32, init_input_dim = 3, init_output_dim=32, num_layers=2).to(device)
# target_net.predict(var_constr_index, constr_var_index, data)
# node = Node({'x1'}, {})
# node.expand(variables, constraints)
# ipdb.set_trace()

# training_ddqn()
training_simple()

