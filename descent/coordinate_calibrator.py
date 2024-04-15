import copy
import math

import numpy as np

from descent.image_graph import ImageGraph
from image_tools.homograph import Homograph
from image_tools.producer import Producer


class Calibrator:
    def __init__(
            self,
            template,
            point_cloud,
            initial_guess: np.array,
            image_graph: ImageGraph = None,
            is_graph: bool = True,
            focal_length: int = 910
    ):
        self.template = template
        self.current_coordinates = initial_guess
        if not is_graph:
            self.producer = Producer(point_cloud, focal_length)
            self.calibrate = self.__calibrate_free
            self.rotate_singular_vector_around()
        else:
            self.graph = image_graph
            self.calibrate = self.__calibrate_for_graph
        self.loss_meter = Homograph()
        self.potential_changes = []
        self.initial_norm = 0.1

    def rotate_singular_vector_around(self):
        for dx in range(10):
            for dy in range(10):
                for dz in range(10):
                    for da in range(1000):
                        self.potential_changes.append(np.array([dx - 5, dy - 5, dz - 5, da / 100 * 2 * np.pi]))

    def normalize_vec_to_norm(self, vec):
        v1 = np.array(list(vec)[:3])
        cnorm = np.linalg.norm(v1)
        v1 = v1 * self.initial_norm / cnorm
        return np.array([*list(v1), vec[-1]])

    def __calibrate_for_graph(self):
        current_loss = math.inf
        new_loss_is_smaller = True

        while new_loss_is_smaller:
            candidates = self.graph.find_k_nearest_neighbors(
                self.current_coordinates, 32
            )
            losses = [self.loss_meter.get_loss(
                x.image, self.template
            ) for x in candidates]
            cand_loss_arr = zip(candidates, losses)
            cand_loss_arr = sorted(cand_loss_arr, key=lambda x: x[1])
            cand, loss = cand_loss_arr[0]
            if loss < current_loss:
                new_loss_is_smaller = True
                current_loss = loss
                self.current_coordinates = cand.coordinates
            else:
                new_loss_is_smaller = False
                break

        return self.current_coordinates

    def __calibrate_free(self):
        loss = self.loss_meter.get_loss(
            self.producer.produce_projection(self.current_coordinates),
            self.template
        )
        while loss > 0.05:
            X_min = self.current_coordinates
            min_loss = copy.deepcopy(loss)
            for delta in self.potential_changes:
                X1 = self.current_coordinates + self.normalize_vec_to_norm(delta)
                new_loss = self.loss_meter.get_loss(
                    self.producer.produce_projection(X1), self.template
                )
                if new_loss < min_loss:
                    X_min = copy.deepcopy(X1)
                    min_loss = copy.deepcopy(new_loss)

            if min_loss == loss:
                break

            self.initial_norm *= min_loss / loss
            self.current_coordinates = X_min
            loss = min_loss

        return self.current_coordinates, loss

