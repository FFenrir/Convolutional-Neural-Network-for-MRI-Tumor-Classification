# model_predictions/predictor.py

from tensorflow import keras
from keras.utils import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
import numpy as np

class MRIModelPrediction:
    def __init__(self, model_path='E:\Convolutional_Neural_Network_MRI_Tumor_Classification\CNN_Model_API\CNN_Model\MRI_CNN_VGG16.keras'):
        self.model = keras.models.load_model(model_path)
        self.class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    def make_prediction(self, image_path):
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        predictions = self.model.predict(image_array)
        decoded_predictions = dict(zip(self.class_labels, predictions[0]))

        return decoded_predictions

# Create an instance of the predictor
model_predictor = MRIModelPrediction()

# Example usage:
# prediction_result = model_predictor.make_prediction('path/to/your/image.jpg')
# print(prediction_result)
