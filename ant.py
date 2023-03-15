import math
import random

import numpy as np


class Ant:
    def __int__(self):
        self.visited = np.empty(dtype=int)
        self.visited = np.append(self.visited, 0)
        self.currentStation = 0
        #self.distance_travelled=0
        return self

    def get_distance_travelled(self, adj_matrix):
        dist=0
        """Return the total distance travelled by the ant"""

    def visit_station(self, pheromone_matrix,Q):
        if random.random() < Q:
            self.visited_attractions.append(self.visit_random_attraction())
        else:
            self.visited_attractions.append(
                self.roulette_wheel_selection(self.visit_probabilistic_attraction(pheromone_matrix,graphMatrix)))
        """Add the next station to the visited array"""

    def visit_random_station(self, adj_matrix):

        """Add the next random station to the visited array"""
    def visit_probablistic_station(self, pheromone_matrix, graphMatrix, count, alpha, beta):
        current_station = self.visited[-1]
        all_stations = np.arange(40)
        bool_arr = np.in1d(all_stations, self.visited)

        possible_cities = all_stations[np.logical_not(bool_arr)]
        possible_indexes = np.empty(dtype=int)
        possible_probabilities = np.empty(dtype=float)
        total_probabilities = 0

        for city in possible_cities:
            possible_indexes = np.append(possible_indexes, city)
            pheromone_on_path = math.pow(pheromone_matrix[current_station][city], alpha)
            heuristic_for_path = math.pow(graphMatrix[current_station][city], beta)
            prob = pheromone_on_path * heuristic_for_path
            possible_probabilities = np.append(possible_probabilities, prob)
            total_probabilities += prob
        possible_probabilities = [probability / total_probabilities for probability in possible_probabilities]
        return [possible_indexes, possible_probabilities]
        """Add the next probablistic station to the visited array"""

    @staticmethod
    def roulette_wheel_selection(self, possible_indexes, possible_probabilities, possible_stations_count):
        slices = np.empty(dtype=int)
        total = 0
        for i in range(possible_stations_count):
            slices = np.append(slices, possible_indexes[i], total, total + possible_probabilities[i])
            total += possible_probabilities[i]
        spin = random.random()
        result = [sl[0] for sl in slices if sl[1] < spin <= sl[2]]
        return result

        """roll the wheel"""
