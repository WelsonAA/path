import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

import random
import bisect

# %%
givenPoints = np.array(
    [[-171.60, 4.00, 0.00],
     [-206.40, 4.20, 0.00],
     [-255.90, 0.20, 0.00],
     [-272.10, -43.90, 0.00],
     [-205.50, -95.00, 0.00],
     [-185.50, -142.40, 0.00],
     [-152.10, -155.00, 0.00],
     [-101.40, -154.70, 0.00],
     [-42.80, -117.20, 0.00],
     [-39.80, -56.80, 0.00],  # 10
     [-40.90, -17.10, 0.00],
     [3.00, -5.70, 0.00],
     [47.80, -5.80, 0.00],
     [89.00, -5.50, 0.00],
     [45.90, -84.90, 0.00],  # 15
     [35.30, 19.30, 0.00],
     [36.30, 71.20, 0.00],
     [41.60, 155.10, 0.00],
     [74.00, 185.20, 0.00],
     [148.10, 170.30, 0.00],  # 20
     [189.20, 52.80, 0.00],
     [174.40, -148.00, 0.00],
     [10.20, -187.90, 0.00],
     [-145.80, -190.90, 8.60],
     [-222.60, 35.10, 10.00],  # 25
     [-119.40, 186.60, 10.00],
     [84.70, 148.10, 0.00],
     [151.10, 116.20, 0.00],
     [151.40, 32.20, 0.00],
     [124.70, 6.90, 0.00],  # 30
     [96.20, -28.60, 0.00],
     [-9.50, -88.30, 0.00],
     [-83.20, -87.70, 0.00],
     [-124.30, -42.40, 0.00],
     [-118.80, 31.10, 0.00],  # 35
     [-124.40, 102.30, 0.00],
     [-80.20, 136.30, 0.00],
     [-16.70, 87.93, 0.00],
     [25.70, 65.40, 0.00],
     [24.60, -30.70, 0.00]
     ])
edges = np.array([[1, 104],
                  [104, 2],
                  [85, 1],
                  [82, 85],
                  [71, 82],
                  [81, 71],
                  [195, 41],
                  [41, 81],
                  [45, 195],
                  [46, 45],
                  [50, 46],
                  [30, 50],
                  [154, 30],
                  [53, 55],
                  [55, 58],
                  [58, 38],
                  [38, 62],
                  [62, 65],
                  [65, 120],
                  [120, 123],
                  [123, 113],
                  [113, 115],
                  [115, 175],
                  [143, 15],
                  [15, 136],
                  [136, 32],
                  [32, 76],
                  [76, 79],
                  [79, 33],
                  [33, 92],
                  [92, 95],
                  [95, 101],
                  [101, 102],
                  [102, 173],
                  [179, 191],
                  [191, 190],
                  [190, 181],
                  [22, 202],
                  [202, 23],
                  [23, 24],
                  [24, 185],
                  [204, 128],
                  [128, 124],
                  [124, 198],
                  [193, 168],
                  [168, 166],
                  [166, 194],
                  [151, 149],
                  [27, 161],
                  [159, 27],
                  [28, 159],
                  [155, 28],
                  [161, 18],
                  [18, 172],
                  [2, 177],
                  [199, 125],
                  [125, 129],
                  [129, 37],
                  [176, 116],
                  [116, 114],
                  [114, 122],
                  [122, 121],
                  [121, 66],
                  [66, 61],
                  [61, 60],
                  [60, 59],
                  [59, 54],
                  [54, 52],
                  [178, 105],
                  [105, 106],
                  [106, 84],
                  [84, 83],
                  [83, 72],
                  [72, 80],
                  [80, 12],
                  [12, 196],
                  [196, 13],
                  [13, 14],
                  [14, 49],
                  [49, 153],
                  [174, 5],
                  [5, 103],
                  [103, 100],
                  [100, 94],
                  [94, 93],
                  [93, 78],
                  [78, 77],
                  [77, 137],
                  [137, 138],
                  [138, 144],
                  [6, 7],
                  [7, 167],
                  [167, 169],
                  [169, 8],
                  [8, 9],
                  [150, 152],
                  [180, 26],
                  [26, 189],
                  [189, 192],
                  [192, 19],
                  [19, 20],
                  [186, 182],
                  [182, 184],
                  [184, 203],
                  [203, 183],
                  [189, 187],
                  [125, 126],
                  [122, 119],
                  [66, 67],
                  [59, 39],
                  [116, 111],
                  [105, 170],
                  [84, 89],
                  [72, 73],
                  [196, 43],
                  [14, 31],
                  [153, 156],
                  [103, 132],
                  [94, 96],
                  [78, 134],
                  [137, 141],
                  [113, 109],
                  [120, 117],
                  [62, 64],
                  [55, 57],
                  [1, 108],
                  [177, 207],
                  [82, 87],
                  [81, 69],
                  [45, 16],
                  [50, 48],
                  [173, 197],
                  [101, 130],
                  [92, 91],
                  [76, 75],
                  [15, 139],
                  [168, 98],
                  [149, 148],
                  [202, 201],
                  [164, 160],
                  [160, 158],
                  [197, 206],
                  [206, 3],
                  [3, 207],
                  [207, 176],
                  [175, 208],
                  [208, 205],
                  [205, 4],
                  [4, 174],
                  [206, 178],
                  [185, 25],
                  [25, 180],
                  [181, 186],
                  [109, 199],
                  [198, 110],
                  [108, 112],
                  [111, 107],
                  [107, 170],
                  [170, 131],
                  [131, 132],
                  [132, 6],
                  [194, 133],
                  [133, 130],
                  [110, 115],
                  [107, 104],
                  [131, 102],
                  [133, 100],
                  [171, 106],
                  [112, 114],
                  [130, 171],
                  [171, 108],
                  [112, 109],
                  [98, 97],
                  [97, 91],
                  [97, 93],
                  [91, 88],
                  [126, 36],
                  [36, 123],
                  [36, 119],
                  [119, 86],
                  [86, 85],
                  [86, 89],
                  [89, 34],
                  [34, 90],
                  [90, 95],
                  [90, 96],
                  [96, 99],
                  [99, 166],
                  [88, 83],
                  [88, 87],
                  [87, 35],
                  [35, 118],
                  [118, 121],
                  [118, 117],
                  [117, 127],
                  [127, 129],
                  [9, 135],
                  [135, 77],
                  [135, 75],
                  [75, 10],
                  [10, 11],
                  [11, 80],
                  [11, 69],
                  [69, 68],
                  [68, 61],
                  [68, 64],
                  [64, 204],
                  [37, 63],
                  [63, 65],
                  [63, 67],
                  [67, 70],
                  [70, 71],
                  [70, 73],
                  [73, 74],
                  [74, 79],
                  [74, 134],
                  [134, 193],
                  [187, 162],
                  [162, 163],
                  [163, 165],
                  [165, 56],
                  [56, 58],
                  [56, 39],
                  [39, 42],
                  [42, 195],
                  [42, 43],
                  [43, 40],
                  [40, 140],
                  [140, 136],
                  [140, 141],
                  [141, 145],
                  [145, 147],
                  [147, 200],
                  [200, 23],
                  [201, 146],
                  [146, 150],
                  [146, 148],
                  [148, 142],
                  [142, 138],
                  [142, 139],
                  [139, 44],
                  [44, 13],
                  [44, 16],
                  [16, 17],
                  [17, 54],
                  [17, 57],
                  [57, 164],
                  [164, 18],
                  [18, 172],
                  [172, 188],
                  [188, 192],
                  [144, 49],
                  [49, 48],
                  [48, 53],
                  [52, 47],
                  [47, 46],
                  [47, 31],
                  [31, 143],
                  [156, 151],
                  [152, 157],
                  [157, 155],
                  [158, 29],
                  [110, 111],
                  [29, 154],
                  [29, 156],
                  [20, 21],
                  [21, 22],
                  [183, 179],
                  [191, 192],
                  [189, 190],
                  [202, 203],
                  [184, 23],

                  [112, 115],
                  [116, 109],
                  [110, 114],
                  [113, 111],

                  [171, 104],
                  [105, 108],
                  [1, 170],
                  [107, 106],

                  [133, 102],
                  [103, 130],
                  [101, 132],
                  [131, 100],

                  [36, 121],
                  [120, 119],
                  [118, 123],
                  [122, 117],

                  [86, 83],
                  [82, 89],
                  [88, 85],
                  [84, 87],

                  [97, 95],
                  [94, 91],
                  [90, 93],
                  [92, 96],

                  [62, 67],
                  [66, 64],
                  [68, 65],
                  [63, 61],

                  [70, 80],
                  [81, 73],
                  [72, 69],
                  [11, 71],

                  [59, 57],
                  [55, 39],
                  [17, 58],
                  [56, 54],

                  [135, 79],
                  [78, 75],
                  [74, 77],
                  [76, 134],

                  [142, 136],
                  [140, 138],
                  [137, 139],
                  [15, 141],

                  [196, 16],
                  [45, 43],
                  [44, 195],
                  [42, 13],

                  [49, 46],
                  [47, 49],
                  [50, 31],
                  [14, 48],

                  [128, 126],
                  [127, 124],

                  [162, 160],
                  [161, 163],

                  [149, 147],
                  [145, 150],

                  [99, 169],
                  [167, 98],

                  [153, 155],
                  [157, 154],

                  [200, 203],
                  [184, 201],

                  [191, 187],
                  [188, 190],

                  [208, 178],
                  [177, 205],
                  ])

intersections = np.load("intersections.npy")

toggle = {
    "36": 117,
    "117": 36,

    "13": 45,
    "45": 13,

    "27": 160,
    "160": 27,

    "15": 138,
    "138": 15,

    "23": 184,
    "184": 23,

    "25": 186,
    "186": 25,

    "34": 91,
    "91": 34,

    "2": 105,
    "105": 2,

    "4": 197,
    "197": 4,

    "8": 168,
    "168": 8,

    "11": 73,
    "73": 11,

    "12": 81,
    "81": 12,

    "16": 42,
    "42": 16,

    "18": 162,
    "162": 18,

    "20": 179,
    "179": 20,

}
# %%
y = givenPoints[:, 0:2]

plt.scatter(y[:, 0], y[:, 1])
plt.show()


# %%
def traslate(point, a=1.45, b=399, c=-1.44, d=308, scale=2):
    x, y = point
    return 2 * int(a * x + b), 2 * int(c * y + d)


# %%
def get_point(s):
    s -= 1
    if (s > 39):
        point = (intersections[s - 40][0], intersections[s - 40][1])
    else:
        point = (givenPoints[s][0], givenPoints[s][1])

    return point


get_point(42)
# %%
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.5
fontColor = (255, 0, 0)
thickness = 1
lineType = 1

img = cv2.imread("img.png")
scale_percent = 200  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)


def plotImg(mainPointsText=True, mainPointsCircles=False, interPointsText=True, interPointsCircles=False,
            showEdges=True, Path=np.array([]), splitPath=False):
    img = cv2.imread("img.png")

    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    c = 1

    for x, y, _ in givenPoints:
        x, y = traslate(point=(x, y))

        bottomLeftCornerOfText = (x, y)

        if mainPointsText == True:
            cv2.putText(img, str(c), bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        if mainPointsCircles == True:
            img = cv2.circle(img, (x, y), 3, (00, 255, 255), 2)
        c += 1

    imgcopy = img
    if showEdges == True:
        for s, d in edges:
            cv2.line(imgcopy, traslate(point=get_point(s)), traslate(point=get_point(d)), (0, 0, 0), 2)

    fif = np.array([2, 4, 8, 138, 81, 117, 73, 13, 91, 42, 23, 25, 27, 18, 20])

    for desierdPoint in fif:
        (x, y) = get_point(desierdPoint)
        x, y = traslate(point=(x, y))
        cv2.circle(imgcopy, (x, y), 20, (00, 255, 255), 3)

    for x, y in intersections:
        x, y = traslate(point=(x, y))

        bottomLeftCornerOfText = (x, y)

        if interPointsCircles == True:
            cv2.circle(imgcopy, (x, y), 3, (00, 255, 255), 2)

        if interPointsText == True:
            cv2.putText(imgcopy, str(c), bottomLeftCornerOfText, font, fontScale * 0.8, (120, 0, 200), thickness,
                        lineType)

        c += 1

    if Path.shape[0] > 0:
        n = Path.shape[0]
        c = 1

        if splitPath == True:
            step = 15

        else:
            step = n

        temp = imgcopy
        imgC = 1

        for j in range(-1, n, step):
            if splitPath == True:
                imgcopy = np.array(temp)
            for i in range(j, min(j + step, n), 1):
                if (i == -1):
                    s = 1
                    d = int(Path[i + 1])
                elif i == n - 1:
                    s = int(Path[i])
                    d = int(1)
                else:
                    s = int(Path[i])
                    d = int(Path[i + 1])

                sx, sy = traslate(point=get_point(s))
                dx, dy = traslate(point=get_point(d))

                center = (int((sx + dx) / 2), int((sy + dy) / 2))
                cv2.putText(imgcopy, str(c), center, font, fontScale, (0, 0, 0), thickness * 2, lineType)

                cv2.line(imgcopy, (sx, sy), (dx, dy), (0, 0, 0), 2)
                c += 1

            if splitPath == True:
                cv2.imwrite('pathes//path' + str(imgC) + '.png', imgcopy)
            imgC += 1

    return imgcopy


img = plotImg(mainPointsText=True, mainPointsCircles=False, interPointsText=True, interPointsCircles=False,
              showEdges=True)
plt.imshow(img[:, :, ::-1])
cv2.imwrite('out.png', img)

# %%
graphList = {}

for s, d in edges:

    dis = math.dist(get_point(s), get_point(d))
    if str(s) in graphList and str(d) in graphList:
        graphList[str(s)].append((d, dis))
        # graphList[str(d)].append((s,dis))

    elif str(s) in graphList:
        graphList[str(s)].append((d, dis))
    # graphList[str(d)] = [(s,dis)]
    elif str(d) in graphList:
        # graphList[str(d)].append((s,dis))
        graphList[str(s)] = [(d, dis)]
    else:
        graphList[str(s)] = [(d, dis)]
    #  graphList[str(d)] = [(s,dis)]

print((graphList["2"]))

# %%
TotalNodeCount = givenPoints.shape[0] + intersections.shape[0]

graphMarix = np.ones(shape=(TotalNodeCount + 1, TotalNodeCount + 1), dtype=float)
graphMarix = graphMarix * float('inf')

for s, d in edges:
    dis = math.dist(get_point(s), get_point(d))

    graphMarix[s][d] = dis
    # graphMarix[d][s] = dis

np.fill_diagonal(graphMarix, 0)
graphMarix[0][0] = float('inf')

print(graphMarix[10][11])

# %%
dp = graphMarix

next = np.ones(shape=(TotalNodeCount + 1, TotalNodeCount + 1), dtype=int)
next = next * -1

for j in range(1, TotalNodeCount + 1):
    for i in range(1, TotalNodeCount + 1):
        if (dp[j][i] != float('inf')):
            next[j][i] = i

        # %%
for k in range(1, TotalNodeCount + 1):
    for j in range(1, TotalNodeCount + 1):
        for i in range(1, TotalNodeCount + 1):
            if (dp[j][k] + dp[k][i] < dp[j][i]):
                dp[j][i] = dp[j][k] + dp[k][i]
                next[j][i] = next[j][k]

            # %%


def getPath(start, end, ans=None):
    start = int(start)
    end = int(end)
    if (ans == None):
        ans = [start]
    else:
        ans.append(start)

    if (start == end):
        return np.array(ans)
    start = next[start][end]
    return getPath(start, end, ans)


# %%
def getDensePath(Path):
    finalpath = getPath(1, Path[0], ans=None)

    n = Path.shape[0]

    for i in range(n - 1):
        finalpath = np.concatenate((finalpath, getPath(Path[i], Path[i + 1], ans=None)), axis=None)

    finalpath = np.concatenate((finalpath, getPath(Path[n - 1], 1, ans=None)), axis=None)

    newfinalPath = np.array([], dtype=int)

    for i in range(finalpath.shape[0]):
        if (newfinalPath.shape[0] == 0 or finalpath[i] != newfinalPath[newfinalPath.shape[0] - 1]):
            newfinalPath = np.insert(newfinalPath, newfinalPath.shape[0], finalpath[i])

    return newfinalPath


getDensePath(np.array([11]))
# %%
img = plotImg(mainPointsText=True, mainPointsCircles=False, interPointsText=True, interPointsCircles=False,
              showEdges=True)

plt.imshow(img[:, :, ::-1])

# %%
ALPHA = 4
BETA = 2
RHO = 0.4
Q = 0.1
# %%
ALPHA = 4
BETA = 2
RHO = 0.4
Q = 0.1

bestcost = 100000
bestpath = None


# %%
class Ant:

    def __init__(self):
        self.visited_stations = np.empty(shape=0, dtype=int)
        self.visited_stations = np.append(self.visited_stations, 1)
        self.currentStation = 0
        self.distance = 0
        # self.way = np.empty(shape=0, dtype=int)

    def __str__(self):

        return f"{self.distance}\n{self.visited_stations}"

    def get_distance_travelled(self):
        dist = 0
        """for i in range(self.visited_stations.size - 1):
            self.way = np.append(self.way, getPath(self.visited_stations[i], self.visited_stations[i + 1]))
        """
        for j in range(self.visited_stations.size - 1):
            dist += dp[int(self.visited_stations[j])][int(self.visited_stations[j + 1])]
        self.distance = dist
        return dist
        """Return the total distance travelled by the ant"""

    def visit_station(self, pheromone_matrix):
        if random.random() < Q:
            x = self.visit_random_station()
            # print(x)
            self.visited_stations = np.append(self.visited_stations, x)
        else:
            possible_indexes, possible_probabilities, possible_stations_count = self.visit_probablistic_station(
                pheromone_matrix, dp)
            y = self.roulette_wheel_selection(
                possible_indexes, possible_probabilities, possible_stations_count)
            # print(y)
            self.visited_stations = np.append(self.visited_stations, y)
        # self.get_distance_travelled()
        """Add the next station to the visited array"""

    def visit_random_station(self):
        all_stations = np.array([2, 4, 8, 138, 81, 117, 73, 91, 42, 13, 23, 25, 27, 18, 20])
        bool_arr = np.in1d(all_stations, self.visited_stations)
        possible_cities = all_stations[np.logical_not(bool_arr)]
        return possible_cities[random.randint(0, len(possible_cities) - 1)]
        """Add the next random station to the visited array"""

    def visit_probablistic_station(self, pheromone_matrix, dp):
        current_station = self.visited_stations[-1]
        temp = np.array(self.visited_stations)
        temp = np.delete(temp, 0)
        all_stations = np.array([2, 4, 8, 138, 81, 117, 73, 91, 42, 13, 23, 25, 27, 18, 20])
        possible_cities = np.empty(shape=0, dtype=int)
        for s in all_stations:
            if s not in temp:
                possible_cities = np.append(possible_cities, s)
        """bool_arr = np.in1d(all_stations, temp)
        possible_cities = all_stations[np.logical_not(bool_arr)]"""
        # print(f"{possible_cities.shape[0]}  {temp.shape[0]}",end='\n')
        possible_indexes = np.empty(shape=0, dtype=int)
        possible_probabilities = np.empty(shape=0, dtype=float)
        total_probabilities = 0

        for city in possible_cities:
            possible_indexes = np.append(possible_indexes, city)
            pheromone_on_path = math.pow(pheromone_matrix[int(current_station)][city], ALPHA)
            heuristic_for_path = math.pow(1 / dp[int(current_station)][city], BETA)
            prob = pheromone_on_path * heuristic_for_path
            possible_probabilities = np.append(possible_probabilities, prob)
            total_probabilities += prob
        possible_probabilities = [probability / (total_probabilities) for probability in
                                  possible_probabilities]
        return [possible_indexes, possible_probabilities, len(possible_cities)]
        """Add the next probablistic station to the visited array"""

    @staticmethod
    def roulette_wheel_selection(possible_indexes, possible_probabilities, possible_stations_count):
        # Calculate the cumulative probabilities
        cumulative_probabilities = [0] * possible_stations_count
        cumulative_probabilities[0] = possible_probabilities[0]
        for i in range(1, possible_stations_count):
            cumulative_probabilities[i] = cumulative_probabilities[i - 1] + possible_probabilities[i]

        # Find the selected item using binary search
        spin = random.uniform(0, cumulative_probabilities[-1])
        index = bisect.bisect_left(cumulative_probabilities, spin)
        # print(possible_indexes[index])
        # Return the selected item's index
        return possible_indexes[index]

    """def roulette_wheel_selection(possible_indexes, possible_probabilities, possible_stations_count):
        # slices = np.empty(shape=0,dtype=int)
        slices = []
        total = 0
        for i in range(possible_stations_count):
            slices.append([possible_indexes[i], total, total + possible_probabilities[i]])
            total += possible_probabilities[i]
        spin = random.random()
        result = [sl[0] for sl in slices if sl[1] < spin <= sl[2]]
        if len(result)==0:
            c=0
        return result"""

    """roll the wheel"""


# %%
class Colony:
    def __init__(self, number_of_ants=3000):
        self.ants = []
        self.phermatrix = np.ones(shape=np.shape(dp), dtype=float)
        self.best_distance = math.inf
        self.best_ant = None
        self.best_distance = 0
        for _ in range(number_of_ants):
            self.ants.append(Ant())

    def appendfinal(self):
        for ant in self.ants:
            ant.visited_stations = np.append(ant.visited_stations, 1)

    def update_phermone_matrix(self, rho, stationsCount):
        for x in range(0, stationsCount):
            for y in range(0, stationsCount):
                self.phermatrix[x][y] *= rho
                for ant in self.ants:
                    if ant.get_distance_travelled() == 0:
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


# %%
cv2.namedWindow('img')
c = 0

best_ant = None
RHO = 0.4
stationsCount = 16

while True:

    global population
    global NormCost

    antC = Colony()
    for r in range(stationsCount - 1):
        antC.move_ants()
    antC.appendfinal()
    antC.update_phermone_matrix(RHO, stationsCount)
    best_ant = antC.get_best()
    global bestcost
    global bestpath
    if best_ant.distance < bestcost:
        bestcost = best_ant.distance
        bestpath = best_ant.visited_stations

    c += 1

    img = plotImg(mainPointsText=True, mainPointsCircles=False, interPointsText=True, interPointsCircles=False,
                  showEdges=False, Path=getDensePath(np.array(bestpath)))
    cv2.putText(img, "iteration: " + str(c) + "  min dis: " + "{:.2f}".format(bestcost), (20, 40), font, fontScale * 2,
                (0, 0, 0), thickness * 5, lineType)

    A, B = dim

    imgcopy = cv2.resize(img, (int(A / 2), int(B / 2)), interpolation=cv2.INTER_AREA)

    cv2.imshow('img', imgcopy)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()

cv2.imwrite('Fullpath.png', img)

img = plotImg(mainPointsText=True, mainPointsCircles=False, interPointsText=True, interPointsCircles=False,
              showEdges=False, Path=getDensePath(np.array(bestpath)), splitPath=True)
# [36 13 27 15 23 25 34  2  4  8 11 12 16 18 20]