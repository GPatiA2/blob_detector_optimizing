from blob_optimizing_ga import BlobGA
from traditionalClassifierLocalStats import TraditionalClassifier
import json
import argparse
import os
import cv2

def options():

    parser = argparse.ArgumentParser(description='Traditional Detector')
    parser.add_argument('--image_path', type=str, default='images', help='Path to dataset')
    parser.add_argument('--tags_path', type=str, default='tags.json', help='Path to tags file')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=10, help='Population size')
    parser.add_argument('--crossover', type=str, default='scattered', help='Crossover type', choices=['single_point', 'two_points', 'uniform', 'scattered'])
    parser.add_argument('--crossover_prob', type=float, default=0.8, help='Crossover probability')
    parser.add_argument('--mutation', type=str, default='random', help='Mutation type', choices=['random', 'swap', 'scramble', 'inversion', 'adaptive'])
    parser.add_argument('--mutation_prob', type=float, default=0.1, help='Mutation probability')
    parser.add_argument('--selection', type=str, default='roulette', help='Selection type', choices=['sss', 'rws', 'sus', 'rank', 'random', 'tournament'])

    args = parser.parse_args()
    print(args)

    return args

def load_dataset(tag_path, im_path):

    dataset = []
    with open(tag_path, 'r') as f:
        tags = json.load(f)

    for it in tags['train'].items():
        path = os.path.join(im_path, it[0])
        im = cv2.imread(path)
        im_t = TraditionalClassifier().transforms()(im)
        dataset.append((im_t, it[1]))

    return dataset

if __name__ == "__main__":

    opt = options()

    dataset = load_dataset(opt.tags_path, opt.image_path)

    print("POSITIVE SAMPLES ", len(list(filter(lambda x: x[1] == 1, dataset))))
    print("NEGATIVE SAMPLES ", len(list(filter(lambda x: x[1] == 0, dataset))))

    mp = TraditionalClassifier().get_max_params()

    print("Dataset loaded")

    ga_instance = BlobGA(dataset, mp, opt.pop_size, opt.generations, opt.crossover, opt.crossover_prob, opt.mutation, opt.mutation_prob, opt.selection)
    
    ga_instance.algorithm.summary()
    
    input("Press enter to run algorithm")

    bs= ga_instance.run()
    
    ga_instance.algorithm.plot_fitness()
    ga_instance.algorithm.plot_genes()
    ga_instance.algorithm.plot_new_solution_rate()

    print(bs)
    print(ga_instance.decode_params(bs[0]))

