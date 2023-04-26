import math

import cv2
import imutils
import numpy as np

img = cv2.imread("img.png", cv2.IMREAD_UNCHANGED)

ix = -1
iy = -1

(imgH, imgW) = img.shape[:2]  # get dimensions
(cX, cY) = (imgW // 2, imgH // 2)  # get center

point_array = np.array([], np.int32)
point_list = []
finished_drawing = False


def cut_out_face(event, x, y, flags, param):
    global ix, iy, point_array, point_list, finished_drawing, img, imgW, imgH
    ix = x
    iy = y

    if event == cv2.EVENT_LBUTTONDBLCLK:

        if len(point_array) > 2 and distance_of_vectors([ix, iy], point_list[0]) < 10:
            point_list.append(point_list[0])
            point_array = np.array(point_list, np.int32)
            cv2.polylines(img, [point_array.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            finished_drawing = True
            print("distance between newPoint and firstPoint: " + str(distance_of_vectors([ix, iy], point_list[0])))
            print("finished drawing!")

        if not finished_drawing:
            cv2.circle(img, (ix, iy), 3, (0, 0, 255), -1)
            point_list.append([ix, iy])
            point_array = np.array(point_list, np.int32)
            cv2.polylines(img, [point_array.reshape((-1, 1, 2))], False, (0, 255, 0), 2)
            print("distance between newPoint and firstPoint: " + str(distance_of_vectors([ix, iy], point_list[0])))

    if finished_drawing:
        mask = np.zeros((imgH, imgW), dtype=np.uint8)
        points = np.array([point_array])
        cv2.fillPoly(mask, points, 255)

        img = cv2.bitwise_and(img, img, mask=mask)

        replace_black_with_white()


def distance_of_vectors(l1, l2):
    l3 = [l2[0] - l1[0], l2[1] - l1[1]]
    l3_length = math.sqrt(l3[0] ** 2 + l3[1] ** 2)
    return l3_length


def replace_black_with_white():
    global imgH, imgW, img
    for i in range(imgH):
        for j in range(imgW):
            if img[i, j].sum() == 0:
                img[i, j] = [255, 255, 255, 0]


cv2.imshow("Display Window", img)

cv2.setMouseCallback("Display Window", cut_out_face)

while 1:
    cv2.imshow("Display Window", img)
    k = cv2.waitKey(1)
    # "e" for rotating img
    if k == ord("e"):
        M = cv2.getRotationMatrix2D((cY, cX), 90, 1.0)
        img = imutils.rotate_bound(img, 90)

        cv2.imshow("Display Window", img)
        cv2.waitKey(0)

    # Save image
    if k == ord("s"):
        cv2.imwrite("newImg.png", img)

    # Exit program
    if k == ord("q"):
        break

cv2.destroyAllWindows()
