import math
import random
import itertools

# Random selection from itertools.combinations(iterable, r)
def random_combination(iterable, r):
    pool = tuple(iterable)
    indices = sorted(random.sample(range(len(pool)), r))
    return tuple(pool[i] for i in indices)

# Check if two tuples are equal
def tuple_equal(a, b):
    return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))

# Generate a CSP instance with forced satisfaction
def generate_forced_satisfaction_instance(k, n, alpha, r, p, file_path):
    d = int(round(pow(n, alpha)))
    m = int(round(r * n * math.log(n)))
    nb = int(round((1 - p) * pow(d, k)))

    with open(file_path, 'w') as file:
        file.write(f"n\td\tm\tk\tnb\talpha\tr\tp\n")
        file.write(f"{n}\t{d}\t{m}\t{k}\t{nb}\t{alpha}\t{r}\t{p}\n")

        rand_sol = [random.randint(0, d - 1) for _ in range(n)]
        file.write(f"{rand_sol}\n")

        for _ in range(m):
            scope = random_combination(range(n), k)
            support = [tuple(rand_sol[j] for j in scope)]

            all_tuples = list(itertools.product(range(d), repeat=k))
            allowed_idxes = random.sample(range(len(all_tuples)), nb - 1)
            for tmp_idx in allowed_idxes:
                chosen_tuple = all_tuples[tmp_idx]
                if not tuple_equal(chosen_tuple, support[0]):
                    support.append(chosen_tuple)
            file.write(f"{scope}|{support}\n")

# Generate a CSP instance without forced satisfaction
def generate_unforced_satisfaction_instance(k, n, alpha, r, p, file_path):
    d = int(round(pow(n, alpha)))
    m = int(round( n * math.log(n)))
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
