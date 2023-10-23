import pygad
from traditionalClassifierLocalStats import TraditionalClassifier
from functools import partial

class BlobGA:

    def __init__(self, dataset, max_vals_params, pop_size, generations, crossover, crossover_prob, mutation, mutation_prob, selection):
        
        self.dataset = dataset

        self.generations = generations

        self.pop_size = pop_size
        
        self.max_vals_params = max_vals_params
        self.gene_space = [{'low': 0.00001, 'high': 1} for i in range(len(max_vals_params))]

        self.crossover = crossover
        self.crossover_prob = crossover_prob

        self.mutation = mutation
        self.mutation_prob = mutation_prob

        self.selection = selection

        def fitness(ga_instance, solution, sol_idx):
            
            classifier = TraditionalClassifier()
            params = self.decode_params(solution)
            try:
                classifier.set_params(params)
            except Exception as e:
                print(e)
                print(params)
                exit()
            
            fp = 0
            tn = 0
            fn = 0
            tp = 0
            
            for it in self.dataset:
                if classifier.predict_step(it[0]) == 1 and it[1] == 1:
                    tp += 1
                elif classifier.predict_step(it[0]) == 0 and it[1] == 0:
                    tn += 1
                elif classifier.predict_step(it[0]) == 1 and it[1] == 0:
                    fp += 1
                elif classifier.predict_step(it[0]) == 0 and it[1] == 1:
                    fn += 1

            prec = (tp + .1) / (tp + fp + .1)
            rec  = (tp + .1) / (tp + fn + .1)
            f1   = 2 * (prec * rec) / (prec + rec)
            
            # print("SOL: ", sol_idx, "FITNESS: ", ret)
            # print("Params: ", params)
            # input()

            return prec
        

        self.algorithm = pygad.GA(num_generations= self.generations,
                                  num_parents_mating=self.pop_size,
                                  fitness_func=fitness,
                                  sol_per_pop=self.pop_size,
                                  stop_criteria="saturate_1",
                                  num_genes=len(self.max_vals_params),
                                  init_range_low=0.00001,
                                  init_range_high=1,
                                  parent_selection_type=self.selection,
                                  keep_parents=1,
                                  save_best_solutions=False,
                                  crossover_type=self.crossover,
                                  crossover_probability=self.crossover_prob,
                                  mutation_type=self.mutation,
                                  random_mutation_max_val=1,
                                  random_mutation_min_val=0.00001,
                                  allow_duplicate_genes=True,
                                  gene_space=self.gene_space,
                                  mutation_probability=self.mutation_prob,
                                  mutation_by_replacement=True,
                                  save_solutions = True,
                                  mutation_percent_genes=10
        )

    def decode_params(self, sol):

        params = [0 for i in range(sol.shape[0])]

        for i in range(len(sol)):
            if sol[i] > 1:
                print(sol[i])
                sol[i] = 1
                print("Changing value to 1")

        # threshold min & max
        # params[0] = sol[0] * self.max_vals_params[0]
        # params[1] = (sol[0] + sol[1]) * self.max_vals_params[1]

        # threshold step
        # params[2] = sol[2] * self.max_vals_params[2]

        # area min & max
        # params[0] = sol[0] * self.max_vals_params[0]
        # params[1] = (sol[1] + sol[0]) * self.max_vals_params[1]

        # circularity min & max
        params[0] = sol[0] * self.max_vals_params[2]
        params[1] = (sol[0] + sol[1]) * self.max_vals_params[1]

        # convexity min & max
        params[2] = sol[2] * self.max_vals_params[4]
        params[3] = (sol[2] + sol[3]) * self.max_vals_params[3]

        # inertia min & max
        params[4] = sol[4] * self.max_vals_params[4]
        params[5] = (sol[4] + sol[5]) * self.max_vals_params[5]

        # min dist between blobs
        # params[8] = sol[8] * self.max_vals_params[8]

        return params


    def run(self):
        self.algorithm.run()
        return self.algorithm.best_solution()