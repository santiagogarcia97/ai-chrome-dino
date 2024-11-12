import datetime
import os
from Game import GameInstance
from GeneticAlgorithm import GeneticAlgorithm
import statistics
import json
import torch
import matplotlib.pyplot as plt

start_time = datetime.datetime.now()
results_path = 'results/' + start_time.strftime('%Y-%m-%dT%H-%M-%S')
os.makedirs(results_path)

population_size = 100 # Cantidad de dinos en cada generacion
mutation_rate = 0.1 # Probabilidad de mutacion
mutation_strength = 0.007 # Fuerza de la mutacion

game_instance = GameInstance(dinos_count=population_size)
game_instance.start_time = start_time
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate, mutation_strength)
results = []

for generation in range(300):
    fitness_scores = genetic_algorithm.evaluate_population(game_instance)
    genetic_algorithm.evolve(fitness_scores)

    best_id = sorted(fitness_scores, key=lambda x: x['fitness'], reverse=True)[0]['id']

    scores = [x['fitness'] for x in fitness_scores]
    results.append(scores)

    best_fitness = max(scores)
    avg_fitness = sum(scores) / len(scores)
    median_fitness = statistics.median(scores)

    game_instance.generation = generation + 1
    game_instance.previous_avg = avg_fitness

    print(f"Generation {generation + 1}: Best fitness = {best_fitness}  Avg = {avg_fitness}  Median = {median_fitness}")

    # Guardo la info de la simulacion en un archivo json dentro de la carpeta results/{start_time}
    # Save results to a json file {generation: [scores]}
    with open(results_path + '/results.json', 'w') as f:
        json.dump(results, f)

    # Save the best model
    torch.save(genetic_algorithm.population[best_id], results_path + '/best_model.pth')

    # Graph best fitness over generations and average fitness over generations in the same graph using matplotlib
    plt.figure()
    plt.plot([max(x) for x in results], label='Mejor puntaje')
    plt.plot([sum(x) / len(x) for x in results], label='Promedio de puntajes')
    plt.xlabel('Generación')
    plt.ylabel('Puntaje')
    plt.legend()
    plt.savefig(results_path + '/graph.png')
    plt.close('all')
    
