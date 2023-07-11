import gym
import pixelate_arena
import time
import pybullet as p
import pybullet_data
import cv2
import numpy as np

if __name__=="__main__":
    env = gym.make("pixelate_arena-v0")
    x=0
    while True:
        p.stepSimulation()
        img = env.camera_feed()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
        # erosion dilation
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(image_mask, kernel, iterations=2)
        dilation = cv2.dilate(erosion, kernel, iterations=2)
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("mask image", image_mask)
        cv2.imshow("img", img)
        cv2.imwrite("sample_arena_img.png", img)
        cv2.waitKey(1)
    time.sleep(1)
