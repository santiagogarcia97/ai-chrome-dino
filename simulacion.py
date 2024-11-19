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

population_size = 300 # Cantidad de dinos en cada generacion
mutation_rate = 0.01 # Probabilidad de mutacion
mutation_strength = 0.007 # Fuerza de la mutacion

game_instance = GameInstance(dinos_count=population_size)
game_instance.start_time = start_time - datetime.timedelta(minutes=43)
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate, mutation_strength)
results = []

# Simulo 300 generaciones
for generation in range(300):
    # Evaluo la poblacion de redes neuronales en el juego
    fitness_scores = genetic_algorithm.evaluate_population(game_instance)

    # Evoluciono la poblacion
    genetic_algorithm.evolve(fitness_scores)

    # ID de la mejor red neuronal de la generacion
    best_id = sorted(fitness_scores, key=lambda x: x['fitness'], reverse=True)[0]['id']

    # Calculo los puntajes de la generacion para mostrarlos en consola
    scores = [x['fitness'] for x in fitness_scores]
    results.append(scores)

    best_fitness = max(scores)
    avg_fitness = sum(scores) / len(scores)
    median_fitness = statistics.median(scores)

    game_instance.previous_avg = avg_fitness
    game_instance.generation += 1

    print(f"Generation {generation + 1}: Best fitness = {best_fitness}  Avg = {avg_fitness}  Median = {median_fitness}")

    # Guardo la info de la simulacion en un archivo json dentro de la carpeta results/{start_time}
    # Save results to a json file {generation: [scores]}
    with open(results_path + '/results.json', 'w') as f:
        json.dump(results, f)

    # Guardo el mejor modelo de la generacion en un archivo
    torch.save(genetic_algorithm.population[best_id], results_path + '/best_model.pth')

    # Grafico los resultados
    plt.figure()
    plt.plot([max(x) for x in results], label='Mejor puntaje')
    plt.plot([sum(x) / len(x) for x in results], label='Promedio de puntajes')
    plt.xlabel('Generación')
    plt.ylabel('Puntaje')
    plt.legend()
    plt.savefig(results_path + '/graph.png')
    plt.close('all')
    
