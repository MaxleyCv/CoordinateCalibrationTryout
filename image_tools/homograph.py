from stitching import Stitcher
import numpy as np
import cv2


class Homograph:

    def __init__(self):
        self.stitcher = Stitcher(detector="orb", confidence_threshold=0.000002)

    def yield_transform(self, field_image, reference_image):
        I = [cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY),
             cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)]

        I2 = []

        for img in I:
            Iy, Ix = np.gradient(img)
            h, w = Iy.shape

            Iq = Ix + Iy

            Iq[Iq < -10] = 255
            Iq[Iq > 10] = 255
            Iq[Iq < 10] = 0

            Ir = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(w):
                for j in range(h):
                    Ir[j][i] = np.array([Iq[j][i], Iq[j][i], Iq[j][i]])

            I2.append(np.uint8(Ir))

        panorama, MH = self.stitcher.stitch(I2)
        return MH[0][0]

    def get_loss(self, image, template):
        try:
            H = self.yield_transform(image, template)
        except Exception as e:
            print(f"Error! {e}")
            return 1.0
        all_white_image = np.ones(image.shape) * 255
        final_image = cv2.warpPerspective(all_white_image, H, (template.shape[1], template.shape[0]))
        # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

        loss = 0

        for i in range(final_image.shape[0]):
            for j in range(final_image.shape[1]):
                if final_image[i][j][0] < 10:
                    loss += 1
        #
        # cv2.imshow("loss", final_image)
        # cv2.waitKey(0)

        return loss / (template.shape[0] * template.shape[1])
