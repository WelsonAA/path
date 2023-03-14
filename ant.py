import numpy as np


class Ant:
    def __int__(self):
        self.visited = np.empty(dtype=int)
        self.visited = np.append(self.visited, 0)

    def get_distance_travelled(self, adj_matrix):
        """Return the total distance travelled by the ant"""

    def visit_station(self, pheromone_matrix):
        """Add the next station to the visited array"""

    def visit_random_station(self,adj_matrix):
        """Add the next random station to the visited array"""\

    def visit_probablistic_attraction(self,pheromone_matrix):
        """Add the next probablistic station to the visited array"""

    def roulette_wheel_selection(self,probs_matrix):
        """roll the wheel"""

