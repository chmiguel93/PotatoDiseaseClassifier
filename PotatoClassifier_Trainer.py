import tensorflow as tf
from tensorflow.keras import layers
import os

standard_size = (256, 256)
batch_sample_size = 64
epochs = 10

potato_data = tf.keras.preprocessing.image_dataset_from_directory(
    "DataSet", #directory where images are stored
    shuffle= True, # take images randomly
    image_size = standard_size, # size of every image of the data
    batch_size = batch_sample_size, # Number of samples(rows)
)

potato_classes = potato_data.class_names #Labels Early_blight, healthy, Late_blight


def Split_DataSet(dataset):
    train_set = int(len(dataset) * 0.8) # 80% for training data.
    val_test_set = test_set = int(len(dataset) * 0.1) # 10% for validation and 10% for test
    
    dataset.shuffle(1000, seed= 10) # Reorder dataset avoiding same entry every call
    training_data = dataset.take(train_set) # Take uses first % set of values
    val_data = dataset.skip(train_set).take(val_test_set)  # Skips avoid using the given number
    test_data = dataset.skip(train_set).skip(val_test_set) # Uses the last set of data
        
    return training_data, val_data, test_data

# Validation data will be used after every batch.
# Test data will be used after all epochs.
training_data, val_data, test_data = Split_DataSet(potato_data)

def Improve_Dataset_Performance(dataset):
    # Cache stores previous used values in memory, prefetch helpps with memory management to improve performance
    return dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)


training_data = Improve_Dataset_Performance(training_data)
val_data = Improve_Dataset_Performance(val_data)
test_data = Improve_Dataset_Performance(test_data)

#CNN requires ndimension of 4 so includes batch and RGB channel
standard_full_size = (batch_sample_size,) + standard_size + (3,)


def Create_TrainingModel():
    # Layer to normalize values
    nom_layer = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(standard_size[0],standard_size[1]), #All images should have same size
        layers.experimental.preprocessing.Rescaling(1.0/255) # Convert to values between 0 and 1
    ])
    
    # One found issue is that we need more images, to solve this we can use data augmentation
    augm_layer = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomZoom(height_factor=(-0.1, -0.1), width_factor=(-0.1, -0.1)),
        layers.experimental.preprocessing.RandomRotation(0.1)
    ])
    
    #Relu converts negative values to 0
    model = tf.keras.Sequential([
        nom_layer,
        augm_layer,
        layers.Conv2D(32, (3,3), activation="relu", input_shape = standard_full_size),
        layers.MaxPooling2D((2,2)), # takes the biggest feature in a filter by making images smaller; hence, improves performance

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(256, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(512, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(1024, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(), #Convert all to array
        layers.Dense(64, activation="relu"),
        layers.Dense(len(potato_classes), activation="softmax") # Classifier layer, softmax converts numbers to probabilities
    ])

    model.build(input_shape = standard_full_size)
    return model


def Train_Model(model):
    model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
    
    model_history = model.fit(training_data, epochs=epochs, batch_size=batch_sample_size,
          validation_data=val_data, verbose=1)  # Verbose=1 shows the training process
    
    model.save("models/PotatoClassifier.h5")
    
    return model


model_path = "models/PotatoClassifier.h5"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = Create_TrainingModel()
    model = Train_Model(model)
    model.summary()
    validation_results = model.evaluate(test_data)
    accuracy = round((validation_results[1] * 100), 2)
    print(f"Accuracy with data never seen before {accuracy}%")