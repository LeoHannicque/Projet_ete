import numpy as np
from typing import List


class DBSCAN:
    def __init__(self, min_points, epsilon):
        self.min_points = min_points
        self.epsilon = epsilon
        self.point_labels = None
        self.cluster_labels = None

    
    def dtw(self,s, t):  #distances between matrices of MFCCs coefficients
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n+1, m+1))
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
    
    
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(s[i-1,:]-t[j-1,:],2)
         
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
        #print(dtw_matrix)
        return dtw_matrix[n,m]/(n+m)


    def fit(self, data):

        labels = [0] * len(data)

        cluster_id = 0

        for point in range(0, len(data)):
            # This loop will ensure we get all the clusters

            if labels[point] != 0:
                # Point is already labeled
                continue

            neighbor_points = self.region_query(data, point)
            #print("le point ",point,"a ",len(neighbor_points),"voisins")

            if len(neighbor_points) < self.min_points:
                # Noise or border point. 
                labels[point] = -1
            else:
                # Point is a core point
                cluster_id += 1
                self.grow_cluster(data, labels, point, neighbor_points, cluster_id)

        return labels

    def grow_cluster(self, data, labels, point, neighbor_points, cluster_id) -> None:

        labels[point] = cluster_id

        i = 0
        while i < len(neighbor_points):

            neighbor_point = neighbor_points[i]

            if labels[neighbor_point] == -1:
                # Relabel the noise point as belonging to this cluster (i.e., border point)
                labels[neighbor_point] = cluster_id

            elif labels[neighbor_point] == 0:
                labels[neighbor_point] = cluster_id

                neighbor_point_neighborhood = self.region_query(data, neighbor_point)

                if len(neighbor_point_neighborhood) >= self.min_points:
                    # This neighborhood has more than min_points,
                    # So add it the neighbor_points FIFO queue
                    neighbor_points += neighbor_point_neighborhood

            i += 1

    def region_query(self, data, this_point) -> List[List]:
        neighbors = []

        for point in range(0, len(data)):
                print(self.dtw(data[this_point],data[point]),this_point,point)
                if self.dtw(data[this_point],data[point]) < self.epsilon:
                #print(self.dtw(data[this_point],data[point]),this_point,point)
                    neighbors.append(point)

        return neighbors
