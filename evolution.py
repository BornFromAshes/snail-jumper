import copy
import random

import numpy as np
import matplotlib.pyplot as plt

import nn
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.most = []
        self.least = []
        self.avg = []

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """

        # TODO (Implement top-k algorithm here)
        players = sorted(players, key=lambda x: x.fitness)
        # TODO (Additional: Implement roulette wheel here)
        # roulette_players = self.roulette_wheel_selection(players, num_players, False)
        # TODO (Additional: Implement SUS here)
        # stochastic_players = self.stochastic_universal_sampling(players, num_players)
        # TODO (Additional: Learning curve)
        # self.learning_curve(players)
        q_tor_players = self.q_tornoment(players, num_players)
        self.learning_curve(players)
        return q_tor_players

    def roulette_wheel_selection(self, population, num_players, replace=True):
        population_fitness = sum([chromosome.fitness for chromosome in population])
        chromosome_probabilities = [chromosome.fitness / population_fitness for chromosome in population]
        chromosome_probabilities = np.array(chromosome_probabilities)
        indexes = np.random.choice(len(population), size=num_players, p=chromosome_probabilities, replace=replace)
        players = [population[indexes[i]] for i in range(num_players)]
        return players

    def stochastic_universal_sampling(self, population, num_players):
        players = []
        population_fitness = sum([chromosome.fitness for chromosome in population])
        chromosome_probabilities = [chromosome.fitness / population_fitness for chromosome in population]
        chromosome_probabilities = np.array(chromosome_probabilities)
        step_size = population_fitness / num_players
        selected = np.random.choice(len(population), p=chromosome_probabilities)
        while len(players) < num_players:
            players.append(population[int(selected)])
            selected += step_size
            if selected > len(population):
                selected %= len(population)
        return players

    def q_tornoment(self, population, num_players):
        players = []
        for i in range(num_players):
            best = -100
            best_index = 0
            for i in range(5):
                ind = random.randint(0, len(population)-1)
                if (best == -100) or population[ind].fitness > best:
                    best = population[ind].fitness
                    best_index = ind
            players.append(population[best_index])
        return players

    def learning_curve(self, population):
        population_fitness = sum([chromosome.fitness for chromosome in population])
        avg = population_fitness / len(population)
        least = 10000000000
        most = -100000000000
        for chromosome in population:
            if chromosome.fitness > most:
                most = chromosome.fitness
            if chromosome.fitness < least:
                least = chromosome.fitness
        self.most.append(most)
        self.least.append(least)
        self.avg.append(avg)
        plt.plot(self.most)
        plt.plot(self.least)
        plt.plot(self.avg)
        plt.legend(["MAX", "MIN", "AVG"])
        plt.show()

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            parents = self.q_tornoment(prev_players, num_players)
            new_players = []
            random.shuffle(parents)
            for i in range(0, num_players, 2):
                child1 = self.clone_player(parents[i])
                child2 = self.clone_player(parents[i + 1])
                child1, child2 = self.crossover(child1, child2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_players.append(child1)
                new_players.append(child2)
            return new_players

    def crossover(self, child1, child2):
        if np.random.rand() > 0.8: return child1, child2
        ch1 = self.vectorized(child1)
        ch2 = self.vectorized(child2)
        index = np.random.randint(0, len(ch1) + 1)
        ch1_crossed = ch1[:index] + ch2[index:]
        ch2_crossed = ch2[:index] + ch1[index:]
        self.unvectorized(ch1_crossed, child1)
        self.unvectorized(ch2_crossed, child2)
        return child1, child2

    def vectorized(self, player):
        nn = player.nn
        vector = []
        for w, b in zip(nn.weights, nn.biases):
            vector.extend(w.reshape(-1))
            vector.extend(b.reshape(-1))
        return vector

    def unvectorized(self, vector, player):
        nn = player.nn
        offset = 0
        for i in range(len(nn.sizes) - 1):
            s = nn.sizes[i] * nn.sizes[i + 1]
            arr = np.zeros(s)
            for j, v in enumerate(vector[offset:offset + s]):
                arr[j] = v
            nn.weights[i] = arr.reshape((nn.sizes[i+1], nn.sizes[i]))
            offset += s
            s = nn.sizes[i + 1]
            arr = np.zeros(s)
            for j, v in enumerate(vector[offset:offset + s]):
                arr[j] = v
            nn.biases[i] = arr.reshape((nn.sizes[i + 1], 1))
            offset += s

    def mutation(self, player):
        vector = self.vectorized(player)
        for i in range(len(vector)):
            if np.random.rand() < 0.04:
                vector[i] += 0.8 * np.random.normal()
        self.unvectorized(vector, player)
        return player

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
