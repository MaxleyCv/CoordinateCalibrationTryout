import unittest
import os

from image_tools.homograph import Homograph

import numpy as np
import cv2

import matplotlib.pyplot as plt


class TestUnitHomograph(unittest.TestCase):
    def test_homograph(self):
        self.loss_meter = Homograph()

        images = os.listdir('test_files')

        l2_distance = []
        losses = []

        get_coords = lambda s: np.array(list(map(float, s[:-4].split('_'))))

        for template in ["1_0_0_0.png"]:
            temp = cv2.imread('test_files/' + template)
            c_temp = get_coords(template)
            print(c_temp)
            c_temp[3] = c_temp[3] / 100
            print("STEP")
            for image in images:
                print(f".\r")
                img = cv2.imread('test_files/' + image)
                c_img = get_coords(image)

                c_img[3] = c_img[3] / 100
                if abs(c_img[3] - c_temp[3]) > 0.1:
                    continue
                dst = np.linalg.norm(c_temp - c_img)
                loss = self.loss_meter.get_loss(img, temp)
                l2_distance.append(dst)
                losses.append(loss)

            print(losses)

        import csv
        with open('data.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(l2_distance)
            writer.writerow(losses)

        plt.scatter(l2_distance, losses)
        plt.show()
