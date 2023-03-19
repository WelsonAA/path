import math
import random

import numpy as np

stationsCount = 40
givenPoints = np.array(
    [[-171.60, 4.00, 0.00],
     [-206.40, 4.20, 0.00],
     [-255.90, 0.20, 0.00],
     [-272.10, -43.90, 0.00],
     [-205.50, -95.00, 0.00],
     [-185.50, -142.40, 0.00],
     [-151.10, -151.00, 0.00],
     [-101.40, -154.70, 0.00],
     [-47.80, -117.20, 0.00],
     [-43.80, -56.80, 0.00],
     [-43.90, -17.10, 0.00],
     [3.00, -2.70, 0.00],
     [47.80, -1.80, 0.00],
     [89.00, -5.50, 0.00],
     [45.90, -84.90, 0.00],
     [31.30, 19.30, 0.00],
     [36.30, 67.20, 0.00],
     [38.60, 155.10, 0.00],
     [74.00, 190.20, 0.00],
     [154.10, 177.30, 0.00],
     [189.20, 52.80, 0.00],
     [174.40, -148.00, 0.00],
     [10.20, -187.90, 0.00],
     [-145.80, -190.90, 8.60],
     [-232.60, 28.10, 10.00],
     [-119.40, 186.60, 10.00],
     [84.70, 144.10, 0.00],
     [148.10, 112.20, 0.00],
     [151.40, 15.20, 0.00],
     [124.70, 1.90, 0.00],
     [96.20, -28.60, 0.00],
     [-9.50, -88.30, 0.00],
     [-83.20, -87.70, 0.00],
     [-124.30, -42.40, 0.00],
     [-121.80, 28.10, 0.00],
     [-124.40, 106.30, 0.00],
     [-80.20, 133.30, 0.00],
     [-20.70, 87.90, 0.00],
     [25.70, 65.40, 0.00],
     [24.60, -30.70, 0.00]
     ])
edges = np.array([[28, 29],
                  [58, 29],
                  [30, 58],
                  [30, 52],
                  [52, 14],
                  [52, 31],
                  [31, 68],
                  [68, 15],
                  [15, 46],
                  [46, 32],
                  [46, 45],
                  [44, 45],
                  [58, 44],
                  [52, 48],
                  [48, 49],
                  [18, 49],
                  [14, 13],
                  [13, 47],
                  [47, 16],
                  [12, 47],
                  [47, 40],
                  [40, 46],
                  [16, 39],
                  [39, 17],
                  [17, 49],
                  [38, 49],
                  [50, 38],
                  [12, 53],
                  [50, 37],
                  [37, 57],
                  [23, 67],
                  [67, 45],
                  [8, 9],
                  [59, 9],
                  [33, 59],
                  [59, 10],
                  [10, 11],
                  [11, 53],
                  [53, 50],
                  [50, 51],
                  [51, 36],
                  [19, 43],
                  [36, 57],
                  [51, 35],
                  [19, 20],
                  [54, 53],
                  [54, 35],
                  [34, 55],
                  [55, 33],
                  [54, 1],
                  [1, 64],
                  [57, 63],
                  [63, 62],
                  [62, 64],
                  [62, 65],
                  [65, 3],
                  [64, 2],
                  [2, 3],
                  [3, 4],
                  [4, 60],
                  [60, 5],
                  [5, 66],
                  [66, 55],
                  [61, 25],
                  [51, 62],
                  [25, 41],
                  [41, 24],
                  [24, 23],
                  [67, 22],
                  [22, 21],
                  [21, 21],
                  [32, 59],
                  [6, 66],
                  [43, 18],
                  [34, 54],
                  [26, 43],
                  [26, 42],
                  [6, 7],
                  [7, 56],
                  [56, 8],
                  [20, 21],
                  [42, 61],
                  [18, 27],
                  [27, 28],
                  [56, 55],
                  [64, 66]
                  ])

intersections = np.array([
    [-229, -143],
    [-210, 156],
    [38, 187],
    [140, -123],
    [35, -141],
    [35, -92],
    [29, -3],
    [96, 72],
    [37, 91],
    [-44, 88],
    [-121, 85],
    [105, -3],
    [-40, 2],
    [-119, -4],
    [-121, -93],
    [-121, -147],
    [-120, 138],
    [154, -6],
    [-44, -89],
    [-244, -86],
    [-243, 82],
    [-186, 88],
    [-173, 137],
    [-186, 2],
    [-247, 76],
    [-184, -93],
    [35, -189],
    [92, -72]
])

fivteenPoints = np.array([[-206.4, 4.2, 0.],
                          [-272.1, -43.9, 0.],
                          [-101.4, -154.7, 0.],
                          [-43.9, -17.1, 0.],
                          [3., -2.7, 0.],
                          [47.8, -1.8, 0.],
                          [45.9, -84.9, 0.],
                          [31.3, 19.3, 0.],
                          [38.6, 155.1, 0.],
                          [154.1, 177.3, 0.],
                          [10.2, -187.9, 0.],
                          [-232.6, 28.1, 10.],
                          [84.7, 144.1, 0.],
                          [-124.3, -42.4, 0.],
                          [-124.4, 106.3, 0.]])

graphList = {}
TotalNodeCount = givenPoints.shape[0] + intersections.shape[0]
graphMatrix = np.ones(shape=(TotalNodeCount, TotalNodeCount), dtype=float)
graphMatrix = graphMatrix * float('inf')
nextstation = np.ones(shape=(TotalNodeCount, TotalNodeCount), dtype=int)
nextstation = nextstation * -1

ALPHA = 4
BETA = 2
RHO = 0.4
Q = 0.1


def get_point(s):
    s -= 1
    if s > 39:
        point = (intersections[s % 40][0], intersections[s % 40][1])
    else:
        point = (givenPoints[s][0], givenPoints[s][1])
    return point


def setup_matrices():
    for s, d in edges:
        dis = math.dist(get_point(s), get_point(d))
        if str(s) in graphList and str(d) in graphList:
            graphList[str(s)].append((d, dis))
            graphList[str(d)].append((s, dis))

        elif str(s) in graphList:
            graphList[str(s)].append((d, dis))
            graphList[str(d)] = [(s, dis)]
        elif str(d) in graphList:
            graphList[str(d)].append((s, dis))
            graphList[str(s)] = [(d, dis)]
        else:
            graphList[str(s)] = [(d, dis)]
            graphList[str(d)] = [(s, dis)]

    for s, d in edges:
        dis = math.dist(get_point(s), get_point(d))
        graphMatrix[s - 1][d - 1] = dis
        graphMatrix[d - 1][s - 1] = dis

    np.fill_diagonal(graphMatrix, 0)
    graphMatrix[0][0] = float('inf')
    dp = graphMatrix

    for j in range(0, TotalNodeCount):
        for i in range(0, TotalNodeCount):
            if dp[j][i] != float('inf'):
                nextstation[j][i] = i
    for k in range(0, TotalNodeCount):
        for j in range(0, TotalNodeCount):
            for i in range(0, TotalNodeCount):
                if dp[j][k] + dp[k][i] < dp[j][i]:
                    dp[j][i] = dp[j][k] + dp[k][i]
                    nextstation[j][i] = nextstation[j][k]


def getPath(start, end, ans=None):
    if ans == None:
        ans = [start]
    else:
        ans.append(start)

    if start == end:
        return np.array(ans)
    start = nextstation[int(start)][int(end)]
    return getPath(start, end, ans)


class Ant:

    def __init__(self):
        self.visited_stations = np.empty(shape=0, dtype=int)
        self.visited_stations = np.append(self.visited_stations, 0)
        self.currentStation = 0
        self.distance = 0
        self.way = np.empty(shape=0, dtype=int)

    def __str__(self):
        return f"{self.distance}\n{self.visited_stations}"

    def get_distance_travelled(self):
        dist = 0
        for i in range(self.visited_stations.size - 1):
            self.way = np.append(self.way, getPath(self.visited_stations[i], self.visited_stations[i + 1]))
        for j in range(self.visited_stations.size-1):
            dist += graphMatrix[int(self.visited_stations[j])][int(self.visited_stations[j+1])]
        self.distance = dist
        return dist
        """Return the total distance travelled by the ant"""

    def visit_station(self, pheromone_matrix):
        if random.random() < Q:
            self.visited_stations = np.append(self.visited_stations, self.visit_random_station())
        else:
            possible_indexes, possible_probabilities, possible_stations_count = self.visit_probablistic_station(
                pheromone_matrix, graphMatrix)
            self.visited_stations = np.append(self.visited_stations,
                                              self.roulette_wheel_selection(
                                                  possible_indexes, possible_probabilities, possible_stations_count))
        #self.get_distance_travelled()
        """Add the next station to the visited array"""

    def visit_random_station(self):
        all_stations = np.arange(40)
        bool_arr = np.in1d(all_stations, self.visited_stations)
        possible_cities = all_stations[np.logical_not(bool_arr)]
        """all_stations = set(range(0, 40))
        possible_stations = all_stations - set(self.visited_stations)"""
        """return np.random.choice(possible_cities)"""
        return random.randint(0, len(possible_cities) - 1)
        """Add the next random station to the visited array"""

    def visit_probablistic_station(self, pheromone_matrix, graphMatrix):
        current_station = self.visited_stations[-1]
        all_stations = np.arange(40)
        bool_arr = np.in1d(all_stations, self.visited_stations)
        possible_cities = all_stations[np.logical_not(bool_arr)]
        possible_indexes = np.empty(shape=0, dtype=int)
        possible_probabilities = np.empty(shape=0, dtype=float)
        total_probabilities = 0

        for city in possible_cities:
            possible_indexes = np.append(possible_indexes, city)
            pheromone_on_path = math.pow(pheromone_matrix[int(current_station)][city], ALPHA)
            heuristic_for_path = math.pow(1/graphMatrix[int(current_station)][city], BETA)
            prob = pheromone_on_path * heuristic_for_path
            possible_probabilities = np.append(possible_probabilities, prob)
            total_probabilities += prob
        possible_probabilities = [probability / total_probabilities for probability in possible_probabilities]
        return [possible_indexes, possible_probabilities, len(possible_cities)]
        """Add the next probablistic station to the visited array"""

    @staticmethod
    def roulette_wheel_selection(possible_indexes, possible_probabilities, possible_stations_count):
        # slices = np.empty(shape=0,dtype=int)
        slices = []
        total = 0
        for i in range(possible_stations_count):
            slices.append([possible_indexes[i], total, total + possible_probabilities[i]])
            total += possible_probabilities[i]
        spin = random.random()
        result = [sl[0] for sl in slices if sl[1] < spin <= sl[2]]
        return result

    """roll the wheel"""


def solve_tsp(it, stationsCount, RHO=0.4):
    best_ant = None
    for i in range(it):
        antC = Colony()
        for r in range(stationsCount - 1):
            antC.move_ants()
        antC.appendfinal()
        antC.update_phermone_matrix(RHO, stationsCount)
        best_ant = antC.get_best()
        print(best_ant)


class Colony:
    def __init__(self, number_of_ants=1000):
        self.ants = []
        self.phermatrix = np.ones(shape=np.shape(graphMatrix), dtype=float)
        self.best_distance = math.inf
        self.best_ant = None
        self.best_distance = 0
        for _ in range(number_of_ants):
            self.ants.append(Ant())


    def appendfinal(self):
        for ant in self.ants:
            ant.visited_stations=np.append(ant.visited_stations,0)

    def update_phermone_matrix(self, rho, stationsCount):
        for x in range(0, stationsCount):
            for y in range(0, stationsCount):
                self.phermatrix[x][y] *= rho
                for ant in self.ants:
                    if ant.get_distance_travelled()==0:
                        continue
                    self.phermatrix[x][y] += 1 / ant.distance

    def move_ants(self):
        for ant in self.ants:
            ant.visit_station(self.phermatrix)

    def get_best(self):
        best = self.ants[0]
        for ant in range(1, len(self.ants)):
            distance_travelled = self.ants[ant].get_distance_travelled()
            if distance_travelled < best.get_distance_travelled():
                best = self.ants[ant]
        self.best_ant = best
        self.best_distance = self.best_ant.distance
        return best


if "__main__" == __name__:
    setup_matrices()
    solve_tsp(10, 10, 0.4)
