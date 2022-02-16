# Multiclass-Food-Classification-and-Calorie-Estimation
### Overview
<p align="justify">
Maintaining a healthy diet is an important goal for many people. The main cause of not maintaining it lies in the imbalance between the amount of calories intake and daily physical activity. One way to achieve this is by tracking the amount of
calories consumed.<br/>
Presently, most of the available calorie estimation tools require the user to enter some information about the food item, and its size. Our proposed approach
alleviates this task of users by automatically estimating the calories from food images.<br/>
We propose a Multiclass food classification using a food image dataset, and then estimating the weight and the calorie content in the respective food item using deep
learning and computer vision techniques.<br/>
</p>

### Dataset
<p align="justify">
The dataset consisted of 2 resources through which the data has been combined.
Both the datasets are available on the internet.
</p>

**Resources**
1. FOODD Dataset
2. ECUST Food Dataset (ECUSTFD) (https://github.com/Liang-yc/ECUSTFD-resized-)

<p align="justify">
The dataset is combined from both resources and contains around 2500 images for
10 classes of food which includes the top and side view of the food. The dataset is
annotated with the density and calorie information per 100 gm serving of the food
item.</p>

**Classes** <br/>
'Apple', 'Banana', 'Beans', 'Boiled Egg', 'Doughnut', 'Grape', 'MoonCake',
'Orange', 'Pasta', 'Qiwi'.

### Data Preprocessing
The data preprocessing steps have been listed out here:
1. Images are resized to 192*192 for the VGG-19 Model, 224*224 for the
Inception V3 Model, and 400*400 for our Proposed CNN Scratch Model.
2. Mean RGB pixel intensity is subtracted from the imagenet dataset.
3. Data augmentation is done by adding images with different orientations and
varying intensities.
4. The images were normalized and standardised for pixel scaling.

### Methodology
The proposed methodology for our Problem Statement has been listed out below.
1. Food Classification using Deep Learning methods: Proposed CNN
Scratch Model and Pretrained models (VGG-19 and Inception-V3) are used to classify the food item.
2. Image Segmentation using Otsu Algorithm (Computer Vision): The food
has been segmented out from the food image in order to find out the exact
food area using the thumb as a calibration object.
3. Volume Estimation: The volume of the food item is calculated using the
shape of the food item.
4. Calorie Calculation: The calories have been calculated using the density of
the food item and calorie per 100gm.

### Web Server
The screenshots from the webserver has been added which has 2 parts:
1. Home
2. Predict

#### Home Page

![Home](https://user-images.githubusercontent.com/43794593/154293242-00f0dcf8-d562-4192-a738-bce59acf7078.png)

#### Predcict Page

![Predict1](https://user-images.githubusercontent.com/43794593/154293267-25c2125a-8746-4fb9-889d-4074a063aef6.png)

![Predict2](https://user-images.githubusercontent.com/43794593/154293296-8f2c0a5e-99d4-4443-b7ce-82a5a7ffc031.png)





