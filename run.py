from flask import Flask,render_template,flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow import keras
from calorie import calories
from model import create_model
import cv2
import numpy as np
import random

IMG_SIZE = 400
LR = 1e-3
no_of_fruits=10

MODEL_NAME = ''

app = Flask('__name__')
UPLOAD_FOLDER = 'static/images/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
       flash('No file part')
       return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        model_save_at=os.path.join("model",MODEL_NAME)
        model = keras.models.load_model(model_save_at)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        labels=list(np.load('label.npy'))        
        img=cv2.imread(image_path)
        img1=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img1 = np.array(img1).reshape(1,400,400,3)
        model_out=model.predict(img1)
        result=np.argmax(model_out)
        name = labels[result].lower()
        cal=str(round(calories(result+1,img),2))
        if name == "apple":
           cal = "Calories: "+cal + " KCal"
           food_name = "Apple"
           desc = "Apples are one of the most popular fruits — and for good reason.\n They’re an exceptionally healthy fruit with many research-backed benefits.\n Apples May Be Good for Weight Loss and your heart. They May Have Prebiotic Effects and Promote Good Gut Bacteria.\n"
        elif name == "banana":
           cal = "Calories: "+cal + " KCal"
           food_name = "Banana"
           desc = "Bananas are one of the most popular fruits worldwide.\n They contain essential nutrients fiber, potassium, folate, and antioxidants, such as vitamin C that can have a protective impact on health.\nBananas are rich in soluble fiber.\n During digestion, soluble fiber dissolves in liquid to form a gel. It’s also what gives bananas their sponge-like texture.\n"
        elif name == "beans":
           cal = "Calories: "+cal + " KCal"
           food_name = "Beans"
           desc = "Green beans are full of fiber, which is an important nutrient for many reasons.\n Soluble fiber, in particular, may help to improve the health of your heart by lowering your LDL cholesterol (bad cholesterol) levels.\n Green beans, string beans, or snap beans are a rich source of vitamins A, C, and K, and of folic acid and fiber.\n"
        elif name == "egg":
           cal = "Calories: "+cal + " KCal"
           food_name = "Egg"
           desc = "Eggs have lots of nutrients, but the health boost you get depends on the way you prepare the meal.\n Hard-boiled eggs are an excellent source of lean protein.\n They are also a source of vitamin A, vitamin D, calcium, and iron. \n"
        elif name == "doughnut":
           cal = "Calories: "+cal + " KCal"
           food_name = "Doughnuts"
           desc = "Doughnuts are usually deep fried from a flour dough, but other types of batters can also be used. Various toppings and flavorings are used for different types, such as sugar, chocolate or maple glazing.\n Doughnuts may also include water, leavening, eggs, milk, sugar, oil, shortening, and natural or artificial flavors.\n The two most common types are the ring doughnut and the filled doughnut, which is injected with fruit preserves (the jelly doughnut), cream, custard, or other sweet fillings.\n"
        elif name == "mooncake":
           cal = "Calories: "+cal + " KCal"
           food_name = "Moon Cake"
           desc = "Mooncakes are round or square in shape, and have a pastry exterior with a dense, sweet filling usually made from lotus paste.\n Many of them also contain a salted duck yolk. Mooncakes are often sliced into wedges, and shared with the family along with tea.\n Mooncakes are composed of two parts, the skin and the filling.\ The skin is made with flour, golden syrup, alkaline water, and cooking oil. The traditional lotus paste filling is made with lotus seeds, with the option to include sesame and nuts.\n"
        elif name == "pasta":
           cal = "Calories: "+cal + " KCal"
           food_name = "Spaghetti Pasta"
           desc = "Spaghetti is a long, thin, solid, cylindrical pasta. It is a staple food of traditional Italian cuisine.\n Like other pasta, spaghetti is made of milled wheat and water and sometimes enriched with vitamins and minerals.\n Spaghetti is made from ground grain (flour) and water. Whole-wheat and multigrain spaghetti are also available.\nRegular spaghetti is fairly neutral, diet-wise, but whole-wheat spaghetti can be a good source of fiber.\n"
        elif name == "grape":
           cal = "Calories: "+cal + " KCal"
           food_name = "Grapes"
           desc = "Grapes offer a wealth of health benefits due to their high nutrient and antioxidant contents.\nCompounds found in grapes may help protect against high cholesterol levels by decreasing cholesterol absorption. \n"
        elif name == "orange":
           cal = "Calories: "+cal+ " KCal"
           food_name = "Orange"
           desc = "Oranges are a treasure trove of nutrients and protective plant compounds, including vitamins, minerals, and antioxidants. \n Oranges are a good source of fiber and a rich source of vitamin C and folate, among many other beneficial nutrients.\n Oranges may benefit heart health, reduce the risk of some chronic diseases, enhance iron absorption, and support a healthy immune response.\n"
        elif name == "qiwi":
           cal = "Calories: "+cal + " KCal"
           food_name = "Qiwi"
           desc = "Qiwis are high in Vitamin C  and dietary fiber and provide a variety of health benefits.\n This tart fruit can support heart health, digestive health, and immunity. \n The kiwi is a healthy choice of fruit and is rich with vitamins and antioxidants.\n Its tart flavor, pleasing texture, and low calorie count make it a delicious and healthy option for snacking, sides, or a unique dessert. \nThe kiwifruit possesses properties that lower blood pressure. By helping to maintain a healthy blood pressure and providing a boost of Vitamin C, the kiwifruit can reduce the risk of stroke  and heart disease.\n"
        else:
            cal = "Calories: "+cal + " KCal"
            food_name = "Pasta"
            desc = "Spaghetti is a long, thin, solid, cylindrical pasta. It is a staple food of traditional Italian cuisine.\n Like other pasta, spaghetti is made of milled wheat and water and sometimes enriched with vitamins and minerals.\n Spaghetti is made from ground grain (flour) and water. Whole-wheat and multigrain spaghetti are also available.\nRegular spaghetti is fairly neutral, diet-wise, but whole-wheat spaghetti can be a good source of fiber.\n"
        return render_template('predict.html', filename=filename,name=food_name,calories=cal,description=desc)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)       

@app.route('/display/<filename>')
def display_image(filename):	
	return redirect(url_for('static', filename='images/' + filename), code=301)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
