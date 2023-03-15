from ant import Ant


class Colony:
    def __int__(self, number_of_ants):
        self.ants = []
        for _ in number_of_ants:
            self.ants.append(Ant())
        return self

    def update_phermone_matrix(self,rho,pheromoneMatrix,stationsCount):
        for x in range(0,stationsCount):
           for y in range(0,stationsCount):
               pheromoneMatrix[x][y]*=rho
               for ant in self.ants:
                   pheromoneMatrix[x][y]+=1/ant.get_distance_travelled()

    def move_ants(self):
        """"""

    def get_best(self):
        best_ant=self.ants[0]
        for ant in range(1,self.ants):
            distance_travelled=ant.get_distance_travelled()
            if distance_travelled<best_ant.get_distance_travelled():
                best_ant=ant
        return best_ant

