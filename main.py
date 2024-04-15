import imutils
import numpy as np
import argparse
import cv2

from stitching import Stitcher
stitcher = Stitcher()

stitcher = Stitcher(detector="orb", confidence_threshold=0.02)


I = [cv2.cvtColor(cv2.imread('testing/test_images_2/screenshot_1.jpg'), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread(
    'testing/test_images_2/s_img_tmp_1.jpg'), cv2.COLOR_BGR2GRAY)]

colorify = lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

I = list(map(lambda x: cv2.resize(x, (640, 480)), I))

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
            # gy, gx = abs(Iy[j][i]), abs(Ix[j][i])
            # gy = 0 if 0 < gy < 100 or 150 < gy else 127
            # gx = 0 if 0 < gx < 100 or 150 < gx else 127
            # Ir[j][i] = [gy + gx, gy + gx, gy + gx]

    I2.append(np.uint8(Ir))

    print(I2[-1])

    #img = Ir

cv2.imshow("h1", I2[1])
cv2.imshow("h2", I2[0])
cv2.waitKey(0)

# I2 = list(map(colorify, I))


print("[INFO] stitching images...")

print("[INFO] stitching images...")
# stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
# (status, stitched) = stitcher.stitch(I2)
#
panorama, MH = stitcher.stitch(I2)

print(MH)

randimg = cv2.warpPerspective(I[0], MH[0][0], dsize=I[0].shape[::-1])

cv2.imshow('stitched', randimg)
cv2.waitKey(0)
