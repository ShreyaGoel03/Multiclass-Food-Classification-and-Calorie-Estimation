import cv2
import numpy as np
from segmentimage import *

density_map = {1:0.96, 2:0.94, 3:0.72, 4:1.03, 5:0.28, 6:0.67, 7:0.52, 8:0.39, 9:0.81, 10:0.94 }
calorie_map = {1:52, 2:89, 3:151, 4:54, 5:412, 6:440, 7:158, 8:69, 9:49, 10:61}
skin_multiplier = 5*2.3

def computerCalories(label, volume): 
	calorie = calorie_map[int(label)]
	density = density_map[int(label)]
	mass = volume*density*1.0
	total_calories = (calorie/100.0)*mass
	return mass, total_calories, calorie

def computeVolume(label, area, skin_area, pix_to_cm_multiplier, fruit):
	fruit_area = (area/skin_area)*skin_multiplier 
	label = int(label)
	volume = 100
	if label == 2: 
		fruit_rect = cv2.minAreaRect(fruit)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = fruit_area/(2.0*height)
		volume = np.pi*radius*radius*height
	else:
		radius = np.sqrt(fruit_area/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius

	return volume

def calories(result,img):
    img_path =img 
    fruit_areas,fruit_bin,fruit_final,skin_areas, fruits, pix_cm = calculateFoodArea(img_path)
    volume = computeVolume(result, fruit_areas, skin_areas, pix_cm, fruits)
    mass, cal, cal100 = computerCalories(result, volume)
    fruit_volumes=volume
    fruit_calories=cal
    fruit_calories_100grams=cal100
    fruit_mass=mass
    return fruit_calories

