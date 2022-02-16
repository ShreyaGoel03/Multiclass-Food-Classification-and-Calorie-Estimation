import cv2
import numpy as np
import os

def calculateFoodArea(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_filter = cv2.medianBlur( img, 5)
    image_threshold = cv2.adaptiveThreshold(image_filter,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
    image_contour = cv2.bitwise_and(image,image,mask = mask)
    hsv_image = cv2.cvtColor(image_contour, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)
    mask_plate = cv2.inRange(hsv_image, np.array([0,0,50]), np.array([200,90,250]))
    mask_not_plate = cv2.bitwise_not(mask_plate)
    fruit_skin = cv2.bitwise_and(image_contour,image_contour,mask = mask_not_plate)
    hsv_image = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv_image, np.array([0,10,60]), np.array([10,160,255]))
    not_skin = cv2.bitwise_not(skin);
    fruit = cv2.bitwise_and(fruit_skin,fruit_skin,mask = not_skin) 
    fruit_black_white = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    fruit_bin = cv2.inRange(fruit_black_white, 10, 255) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)
    image_threshold = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask_fruit = cv2.dilate(mask_fruit,kernel,iterations = 1)
    fruit_final = cv2.bitwise_and(image,image,mask = mask_fruit)
    image_threshold = cv2.adaptiveThreshold(mask_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_areas = sorted(contours, key=cv2.contourArea)
    fruit_contour = largest_areas[-2]
    fruit_area = cv2.contourArea(fruit_contour)
    skin2 = skin - mask_fruit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    skin_e = cv2.erode(skin2,kernel,iterations = 1)
    image_threshold = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_skin = np.zeros(skin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask_skin, [largest_areas[-2]], 0, (255,255,255), -1)
    skin_rect = cv2.minAreaRect(largest_areas[-2])
    skin_skin_box = cv2.boxPoints(skin_rect)
    skin_skin_box = np.int0(skin_skin_box)
    mask_skin2 = np.zeros(skin.shape, np.uint8)
    cv2.drawContours(mask_skin2,[skin_skin_box],0,(255,255,255), -1)
    pix_height = max(skin_rect[1])
    pix_to_cm_multiplier = 5.0/pix_height
    skin_area = cv2.contourArea(skin_skin_box)
    return fruit_area,fruit_bin ,fruit_final,skin_area, fruit_contour, pix_to_cm_multiplier



