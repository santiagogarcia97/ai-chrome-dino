from Game import GameInstance
from GeneticAlgorithm import GeneticAlgorithm

population_size = 100
mutation_rate = 0.01
mutation_strength = 0.005

game_instance = GameInstance(dinos_count=population_size)
genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate, mutation_strength)

results = []

for generation in range(300):
    fitness_scores = genetic_algorithm.evaluate_population(game_instance)
    genetic_algorithm.evolve(fitness_scores)
    best_fitness = max(fitness_scores, key=lambda x: x['fitness'])['fitness']
    results.append(best_fitness)
    print(f"Generation {generation + 1}: Best fitness = {best_fitness}  Avg fitness = {sum(
        x['fitness'] for x in fitness_scores) / len(fitness_scores)} Mediana = {sorted([x['fitness'] for x in fitness_scores])[len(fitness_scores) // 2]}")
    
    #print(results)
