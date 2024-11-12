import random
import torch
from DinoNet import DinoNet
import copy


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, mutation_strength):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = [DinoNet() for _ in range(population_size)]

    # Evaluo la poblacion de redes neuronales en el juego, para cada frame, obtengo el estado del dinosaurio y le pido una accion a realizar
    def evaluate_population(self, game_instance):
        # Reinicio el juego
        game_instance.restart()

        # Resultados de la evaluacion de la poblacion {id, puntaje obtenido}
        fitness_scores = []

        # Mientras que haya al menos un dinosaurio vivo
        all_dinos_dead = False
        while not all_dinos_dead:

            # Lista de acciones a realizar por cada red neuronal de la poblacion, cada red neuronal se corresponde con un dinosaurio
            actions = []

            # Para cada red neuronal de la poblacion, si el dinosaurio esta vivo, obtengo su estado y le pido una accion a realizar
            for i in range(len(game_instance.dinos)):
                if not game_instance.dinos[i].is_alive:
                    actions.append("NONE")
                else:
                    state = game_instance.dinos[i].get_state(
                        game_instance.obstacles, game_instance.game_speed)
                    actions.append(self.get_action(self.population[i], state))

            # Realizo un paso del juego con las acciones obtenidas
            dinos = game_instance.play_step(actions)
            all_dinos_dead = all(not dino.is_alive or (
                dino.is_alive and dino.points > 5000) for dino in dinos)

        # Obtengo los puntajes de cada dinosaurio
        for dino in game_instance.dinos:
            fitness_scores.append({"id": dino.id, "fitness": dino.points})

        return fitness_scores

    # A partir de un estado, obtengo la accion a realizar por el dinosaurio
    def get_action(self, network, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        output = network(state_tensor)

        # Obtengo la probabilidad de cada clase, puede ser 0, 1 o 2
        probabilities = torch.softmax(output, dim=0)
        # Obtengo la clase con mayor probabilidad
        predicted_class = torch.argmax(probabilities).item()

        if predicted_class == 0:
            return "CROUCH"
        elif predicted_class == 1:
            return "JUMP"
        else:
            return "NONE"

    # A partir de los puntajes de fitness de cada red neuronal, selecciono un porcentaje de las mejores redes neuronales
    def selection(self, fitness_scores, percentage=0.1):
        # Sort fitness scores in descending order
        fitness_scores = sorted(
            fitness_scores, key=lambda x: x["fitness"], reverse=True)
        n = int(self.population_size * percentage)
        selected_networks = []
        for i in range(n):
            selected_networks.append(self.population[fitness_scores[i]["id"]])

        return selected_networks
    
    # Evolucion de la poblacion de redes neuronales
    def evolve(self, fitness_scores):
        new_population = []

        # Selecciono el 5% de los mejores performers que van a pasar directamente a la siguiente generacion
        # y van a ser los padres de la nueva generacion
        parents = self.selection(fitness_scores, percentage=0.05)
        for parent in parents:
            new_population.append(copy.deepcopy(parent))

        # Obtengo la red neuronal con mejor puntaje
        best_performer = parents[0]

        # Un 20% va a ser copias del mejor performer
        #for _ in range(int(self.population_size * 0.2)):
            #new_population.append(copy.deepcopy(best_performer))

        # un 30% va a ser generado mutando el mejor performer
        for _ in range(int(self.population_size * 0.3)):
            new_population.append(self.mutate(best_performer))

        # un 20% va a ser generado cruzando los padres seleccionados
        for _ in range(int(self.population_size * 0.40)):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child))

        # un 20% va a ser generado cruzando un padre seleccionado con una red neuronal nueva
        for _ in range(int(self.population_size * 0.2)):
            random_parent = random.choice(parents)
            new_dino = DinoNet()
            child = self.crossover(random_parent, new_dino)
            new_population.append(child)

        # Completo el % restante con redes neuronales nuevas
        while len(new_population) < self.population_size:
            new_population.append(DinoNet())

        self.population = new_population


    # Mutacion de una red neuronal
    def mutate(self, network):
        new_network = DinoNet()
        for (new_param, old_param) in zip(new_network.parameters(), network.parameters()):
            # Apply mutation with a certain probability
            if random.random() < self.mutation_rate:
                # Add a random value scaled by the mutation strength
                new_param.data = old_param.data + \
                    torch.randn_like(old_param.data) * self.mutation_strength
            else:
                new_param.data = old_param.data.clone()
        return new_network

    # Cruzamiento de dos redes neuronales para obtener una nueva red neuronal
    def crossover(self, network1, network2):
        child_network = DinoNet()
        for (child_param, param1, param2) in zip(child_network.parameters(), network1.parameters(), network2.parameters()):
            # Randomly choose weights/biases from either parent
            mask = torch.randint(0, 2, child_param.shape, dtype=torch.bool)
            child_param.data = torch.where(
                mask, param1.data, param2.data)
        return child_network