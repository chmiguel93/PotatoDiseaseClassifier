import tensorflow as tf
import numpy as np
import os
import PotatoClassifier_Trainer

standard_size = PotatoClassifier_Trainer.standard_size
potato_classes = PotatoClassifier_Trainer.potato_classes

def Load_model():
    model_path = "models/PotatoClassifier.h5"

    if os.path.exists(model_path):
        loaded_model = tf.keras.models.load_model(model_path)
    else:
        loaded_model = PotatoClassifier_Trainer.CreateTrainingModel()
        loaded_model = PotatoClassifier_Trainer.TrainModel(model)

    return loaded_model

model = Load_model()

def Predict_Image(potato_image):
    # Since we need to predict only one img, batch should be none (,256,256,3)
    predicted_values = model.predict(potato_image[None, :, :])
    predicted_index = np.argmax(predicted_values[0])  # Get max probability from predicted values
    return PotatoClassifier_Trainer.potato_classes[predicted_index]