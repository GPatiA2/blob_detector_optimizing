import pygad
from traditionalClassifierLocalStats import TraditionalClassifier

class BlobGA:

    def __init__(self, dataset, pop_size, max_vals_params, generations, crossover, crossover_prob, mutation, mutation_prob, selection):
        
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

        self.algorithm = pygad.GA(num_generations= self.generations,
                                  num_parents_mating=2,
                                  fitness_func=self.fitness_func,
                                  sol_per_pop=self.pop_size,
                                  stop_criteria="saturate_1",
                                  num_genes=len(self.max_vals_params),
                                  init_range_low=0,
                                  init_range_high=1,
                                  parent_selection_type=self.selection,
                                  keep_parents=1,
                                  save_best_solutions=True,
                                  crossover_type=self.crossover,
                                  crossover_probability=self.crossover_prob,
                                  mutation_type=self.mutation,
                                  allow_duplicate_genes=True,
                                  mutation_probability=self.mutation_prob,
        )

    def fitness_func(self, sol, sol_idx): 

        classifier = TraditionalClassifier()
        params = [self.max_vals_params[i] * sol[i] for i in range(len(sol))]
        new_params, penalty = self.correct_params(params)
        classifier.set_params(params)

        hits = 0
        for it in self.dataset:
            if classifier.predict_step(it) == it[1]:
                hits += 1

        return hits / len(self.dataset)
    
    def correct_params(self, params):
         
    
    def run(self):
        self.algorithm.run()
        return (self.algorithm.best_solution(), self.algorithm.best_solutions)