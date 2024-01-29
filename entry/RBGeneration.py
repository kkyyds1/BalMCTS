import math
import random
import itertools
import ipdb
from entry.parameter import *
# Random selection from itertools.combinations(iterable, r)
def random_combination(iterable, r):
    pool = tuple(iterable)
    indices = sorted(random.sample(range(len(pool)), r))
    return tuple(pool[i] for i in indices)

# Check if two tuples are equal
def tuple_equal(a, b):
    return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))

# Generate a COP instance with forced satisfaction
def generate_forced_optimization_instance(k, n, alpha, r, p, file_path):
    d = math.ceil((pow(n, alpha)))
    # m = math.ceil(r * n * math.log(n))
    m = [(n1, n2) for n1 in range(n) for n2 in range(n) if n1 != n2]
    nb = math.ceil((1 - p) * pow(d, k))

    with open(file_path, 'w') as file:
        file.write(f"n\td\tm\tk\tnb\talpha\tr\tp\n")
        file.write(f"{n}\t{d}\t{len(m)}\t{k}\t{nb}\t{alpha}\t{r}\t{p}\n")

        rand_sol = [random.randint(0, d - 1) for _ in range(n)]
        file.write(f"{rand_sol}\n")

        exist = []
        optimal_solution = {}
        for n1, n2 in m:
            if (n1, n2) in exist:
                continue
            exist.append((n1, n2))
            exist.append((n2, n1))
            temp_dict = {}
            scope = n1, n2
            # scope = random_combination(range(n), k)
            temp = tuple(rand_sol[j] for j in scope)
            support = [(temp[0], temp[1], random.randint(min_cost, max_cost))]
            sym_support = [(temp[1], temp[0], support[0][2])]
            optimal_solution[scope] = support[0][2]
            
            all_tuples = list(itertools.product(range(d), repeat=k))
            allowed_idxes = random.sample(range(len(all_tuples)), nb - 1)
            for tmp_idx in allowed_idxes:
                temp = all_tuples[tmp_idx]
                if (temp[0],temp[1]) in temp_dict:
                    continue
                chosen_tuple = (temp[0], temp[1], random.randint(optimal_solution[scope], max_cost))
                temp_dict[temp[0], temp[1]] = chosen_tuple[2]
                if not tuple_equal(chosen_tuple, support[0]):
                    support.append(chosen_tuple)
                    sym_support.append((chosen_tuple[1], chosen_tuple[0], chosen_tuple[2]))
            file.write(f"{(n1, n2)}|{support}\n")
            file.write(f"{(n2, n1)}|{sym_support}\n")


def generate_forced_satisfaction_instance(k, n, alpha, r, p, file_path):
    d = math.ceil((pow(n, alpha)))
    m = math.ceil(r * n * math.log(n))
    nb = math.ceil((1 - p) * pow(d, k))

    with open(file_path, 'w') as file:
        file.write(f"n\td\tm\tk\tnb\talpha\tr\tp\n")
        file.write(f"{n}\t{d}\t{m}\t{k}\t{nb}\t{alpha}\t{r}\t{p}\n")

        rand_sol = [random.randint(0, d - 1) for _ in range(n)]
        file.write(f"{rand_sol}\n")

        for _ in range(m):
            scope = random_combination(range(n), k)
            temp = tuple(rand_sol[j] for j in scope)
            support = [temp]
            all_tuples = list(itertools.product(range(d), repeat=k))
            allowed_idxes = random.sample(range(len(all_tuples)), nb - 1)
            for tmp_idx in allowed_idxes:
                temp = all_tuples[tmp_idx]
                chosen_tuple = temp
                if not tuple_equal(chosen_tuple, support[0]):
                    support.append(chosen_tuple)
            file.write(f"{scope}|{support}\n")

# Generate a CSP instance without forced satisfaction
def generate_unforced_satisfaction_instance(k, n, alpha, r, p, file_path):
    d = int(round(pow(n, alpha)))
    m = int(round(r * n * math.log(n)))
    nb = int(round((1 - p) * pow(d, k)))

    with open(file_path, 'w') as file:
        file.write(f"n\td\tm\tk\tnb\talpha\tr\tp\n")
        file.write(f"{n}\t{d}\t{m}\t{k}\t{nb}\t{alpha}\t{r}\t{p}\n")
        file.write("\n")

        for _ in range(m):
            scope = random_combination(range(n), k)
            all_tuples = list(itertools.product(range(d), repeat=k))
            allowed_idxes = random.sample(range(len(all_tuples)), nb - 1)
            support = [all_tuples[tmp_idx] for tmp_idx in allowed_idxes]
            file.write(f"{scope}|{support}\n")

# Generate multiple CSP instances
def generate_instance_files(k, n, alpha, r, p, file_path, num_instances, if_sat=True):
    for i in range(num_instances):
        instance_file_path = f"{file_path}{i}.txt"
        if if_sat:
            generate_forced_satisfaction_instance(k, n, alpha, r, p, instance_file_path)
        else:
            generate_unforced_satisfaction_instance(k, n, alpha, r, p, instance_file_path)

# if __name__=='__main__':
#     generate_instance_files(2, 20, 0.7, 3, 0.21, "problems/", 10)
