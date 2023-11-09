'''A script to simulate population divergence, visualized through PCA on SNPs'''

import random
import warnings
import os
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

def generate_population(num_individuals: int = 1000, num_snps: int = 100):
    '''Generates a starting population'''
    allele_frequencies = [round(random.uniform(0.5,1),2) for i in range(num_snps)]
    population =[]
    for _ in range(num_individuals):
        individual_row = []
        for allele_frequency in allele_frequencies:
            individual_row.append((0 if random.random() < allele_frequency else 1, 
                                   0 if random.random() < allele_frequency else 1))
        population.append(individual_row)
    return np.array(population)

def split_population(population_matrix: np.array, k: int = 2):
    '''Splits population into k groups'''
    return np.split(population_matrix, k)

def generate_pca(population_matrices_lst: list, generation_count: int, directory_path: str):
    '''Generates a principal component analysis plot of a population.
        population: numpy array'''

    population_matrix = np.concatenate(population_matrices_lst)
    population_matrix = np.count_nonzero(population_matrix, axis=2)
    num_individuals = population_matrix.shape[0]
    pca = decomposition.PCA(n_components=2)
    pca.fit(population_matrix)
    to_plot = pca.transform(population_matrix)
    to_plot = pd.DataFrame(to_plot, columns = ['PCA1','PCA2'])

    # Plotting
    labels = [[str(k)] * (num_individuals // len(population_matrices_lst)) for k in range(len(population_matrices_lst)) ]
    labels = [item for sublist in labels for item in sublist]
    to_plot['group'] = pd.Categorical(labels)

    plt.clf()  # Clears the previous plot
    sns.scatterplot(x=to_plot.PCA1, y=to_plot.PCA2, hue=to_plot.group)
    plt.savefig(f'./{directory_path}/{str(generation_count).zfill(2)}.jpg')

def sim_breeding(population_matrix: np.array):
    '''Simulates one generation of random breeding, returns an array with new genotypes'''
    num_individuals = population_matrix.shape[0]
    allele_frequencies =  [round(1 - (sum(x) / (num_individuals * 2)),5)  for x in np.sum(population_matrix, axis=0)]
    population =[]
    for _ in range(num_individuals):
        individual_row = []
        for allele_frequency in allele_frequencies:
            individual_row.append((0 if random.random() < allele_frequency else 1, 
                                   0 if random.random() < allele_frequency else 1))
        population.append(individual_row)
    return np.array(population)


def sim_main(generations: int = 100, num_individuals: int = 100, num_snps: int = 100, k:int = 2, stages: list = None):
    '''The main show!'''
    print(f'Simulating {generations} generations with {k} populations each with {num_individuals} individuals with {num_snps} snps')
    if stages is None:
        stages = [generations//2, generations]

    # Creating directory path
    try:
        directory_path = f'./{generations}-{num_individuals}-{num_snps}-{k}'
        os.makedirs(directory_path)
    except FileExistsError:
        try:
            directory_path = f'./{generations}-{num_individuals}-{num_snps}-{k} (1)'
            os.makedirs(directory_path)
        except FileExistsError:
            index = 2
            while True:
                try:
                    os.makedirs(directory_path)
                except FileExistsError:
                    directory_path = f'{directory_path[:-4]} ({index})'
                    index += 1
                else:
                    break

    gen0 = generate_population(num_individuals * k, num_snps)
    population_matrices_lst = split_population(gen0, k)
    generate_pca(population_matrices_lst, 0, directory_path)

    for generation_count in range(1,generations+1):
        population_matrices_lst = [sim_breeding(x) for x in population_matrices_lst]
        if generation_count in stages:
            generate_pca(population_matrices_lst, generation_count, directory_path)
        print(f'Generation {generation_count} of {generations} complete.')

    print(f'Simulation complete. Plots can be found within {directory_path}')
