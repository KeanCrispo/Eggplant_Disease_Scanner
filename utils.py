# Importing Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image

# Cleaning image    
def clean_image(image):
    image = np.array(image)
    image = np.array(Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS))
    image = image[np.newaxis, :, :, :3]
    return image

def get_prediction(model, image):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test = datagen.flow(image)
    predictions = model.predict(test)
    predictions_arr = np.argmax(predictions)
    return predictions, predictions_arr

# Making the final results with custom diagnosis links
def make_results(predictions, predictions_arr):
    result = {}
    if int(predictions_arr) == 0:
        result = {
            "status": " is Healthy ",
            "prediction": f"{int(predictions[0][0].round(2)*100)}%",
            "link": "https://thisismygarden.com/2021/05/growing-eggplant/"
        }
    elif int(predictions_arr) == 1:
        result = {
            "status": ' has Multiple Diseases ',
            "prediction": f"{int(predictions[0][1].round(2)*100)}%",
            "link": "https://gardeningtips.in/21-common-eggplant-problems-how-to-fix-them-solutions-and-treatment"
        }
    elif int(predictions_arr) == 2:
        result = {
            "status": ' has fruit rot ',
            "prediction": f"{int(predictions[0][2].round(2)*100)}%",
            "link": "https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/eggplant/fruitspots.html"
        }
    elif int(predictions_arr) == 3:
        result = {
            "status": ' has leaf bright ',
            "prediction": f"{int(predictions[0][3].round(2)*100)}%",
            "link": "https://www.gardeningknowhow.com/edible/vegetables/eggplant/early-blight-on-eggplants.htm"
        }
    return result
