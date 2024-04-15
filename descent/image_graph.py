import copy

import numpy as np


class ImageGraph:
    def __init__(self):
        self.node_list = []
        self.connection_list = []

    def add_node(self, image, coords):
        self.node_list.append(
            Node(
                image, *coords
            )
        )

    def find_k_nearest_neighbors(self, coords, k = 10):
        new_list = copy.copy(self.node_list)
        new_list.sort(
            key=lambda node: node.distance(coords)
        )
        return new_list[:min(len(new_list), k)]


class Node:
    def __init__(self, image, x, y, z, ang):
        self.image = image
        self.coordinates = np.array([x, y, z, ang])

    def distance(self, coordinates):
        return np.linalg.norm(
            np.array(list(self.coordinates)[:3])
            - np.array(list(coordinates)[:3]))
