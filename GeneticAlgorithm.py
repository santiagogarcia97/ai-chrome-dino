import random
import torch
from DinoNet import DinoNet

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, mutation_strength):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = [DinoNet() for _ in range(population_size)]

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
            all_dinos_dead = all(not dino.is_alive for dino in dinos)

        # Obtengo los puntajes de cada dinosaurio
        for dino in game_instance.dinos:
            fitness_scores.append({"id": dino.id, "fitness": dino.points})

        return fitness_scores


    # A partir de un estado, obtengo la accion a realizar por el dinosaurio
    def get_action(self, network, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        output = network(state_tensor)
        # Select action with highest probability
        results = torch.softmax(output, dim=0)
        predicted_class = torch.argmax(results).item()

        if predicted_class == 0:
            return "CROUCH"
        elif predicted_class == 1:
            return "JUMP"


    def selection(self, fitness_scores):
        # Sort fitness scores in descending order
        fitness_scores = sorted(
            fitness_scores, key=lambda x: x["fitness"], reverse=True)
        # Cantidad a seleccionar (10%)
        n = int(self.population_size * 0.1)
        selected_networks = []
        for i in range(n):
            selected_networks.append(self.population[fitness_scores[i]["id"]])

        return selected_networks

    def crossover(self, parent1, parent2):
      # Create a new child by copying the structure of parent1
      child = DinoNet()
      child.load_state_dict(parent1.state_dict())
      
      # Iterate over parameters of both parents and copy to the child
      for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
          # Create a mask to randomly select weights from parent1 or parent2
          crossover_mask = torch.rand(child_param.shape) < 0.5
          # Apply mask: where crossover_mask is True, use parent1's weight; otherwise, use parent2's weight
          child_param.data = torch.where(crossover_mask, p1_param.data, p2_param.data)
      
      return child

    def mutation(self, network):
      for param in network.parameters():
          if param.requires_grad:  # Only mutate parameters that require gradients
              # Generate a mask for where mutations should occur
              mutation_mask = (torch.rand(param.shape) < self.mutation_rate).float()
              
              # Generate random mutation values (Gaussian noise)
              mutation_values = torch.randn(param.shape) * self.mutation_strength
              
              # Apply mutation where the mask is 1
              param.data += mutation_mask * mutation_values

    def evolve(self, fitness_scores):
        new_population = []

        # Selecciono el 10% de los mejores performers
        parents = self.selection(fitness_scores)
        for parent in parents:
            new_dino = DinoNet()
            new_dino.load_state_dict(parent.state_dict())
            new_population.append(new_dino)

        best_performer = parents[0]

        # un 30% va a ser generado mutando el mejor performer
        for _ in range(int(self.population_size * 0.3)):
            child = DinoNet()
            child.load_state_dict(best_performer.state_dict())
            self.mutation(child)
            new_population.append(child)

        # un 50% va a ser generado cruzando los padres seleccionados
        for _ in range(int(self.population_size * 0.5)):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            self.mutation(child)
            new_population.append(child)

        # Completo el 5% restante con redes neuronales nuevas
        while len(new_population) < self.population_size:
            new_population.append(DinoNet())

        self.population = new_population