# Modelling-MNIST-using-Triplet-Neural-Networks--
Modeling the MNIST dataset using a Triplet Neural Network (TNN) is an interesting approach to learning embeddings for handwritten digits. A triplet network is a type of neural network designed to learn to distinguish between similar and dissimilar items by embedding them into a space where the distance between similar items is small and the distance between dissimilar items is large. This is typically used for tasks like Face Verification or One-shot Learning.
Key Components of a Triplet Neural Network:

    Anchor: A reference input image (e.g., a digit image from MNIST).
    Positive: Another image from the same class as the anchor.
    Negative: An image from a different class than the anchor.

The goal of a Triplet Loss Function is to minimize the distance between the anchor and positive pairs, while maximizing the distance between the anchor and negative pairs in the learned embedding space.
Steps to Implement a Triplet Neural Network on MNIST:

    Dataset Preparation: Prepare triplets from MNIST (anchor, positive, negative).
    Model Architecture: Define a shared neural network model that computes embeddings for each input.
    Loss Function: Use Triplet Loss to ensure the model learns embeddings that respect the similarity constraints.
    Training: Train the model to minimize the triplet loss.

Below is the full code to implement this using TensorFlow/Keras.
1. Import Libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import random
import matplotlib.pyplot as plt

2. Load and Preprocess the MNIST Dataset

First, we load the MNIST dataset and normalize the pixel values.

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape to include the channel dimension (28x28x1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

3. Create the Triplet Loss Function

The Triplet Loss function ensures that the distance between the anchor and positive images is smaller than the distance between the anchor and negative images by a margin.

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        # Split the predicted embeddings into anchor, positive, and negative embeddings
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Compute the distance between anchor and positive embeddings
        pos_distance = K.sum(K.square(anchor - positive), axis=1)

        # Compute the distance between anchor and negative embeddings
        neg_distance = K.sum(K.square(anchor - negative), axis=1)

        # Return the triplet loss with the margin
        return K.mean(K.maximum(pos_distance - neg_distance + margin, 0.0))
    return loss

4. Build the Embedding Model

We define a simple CNN model for generating embeddings. This network is shared across the anchor, positive, and negative images.

def create_embedding_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    return model

5. Create the Triplet Network

Now we create the Triplet Network by feeding the anchor, positive, and negative images into the same embedding model.

def create_triplet_network():
    embedding_model = create_embedding_model()

    # Define the input layers for the anchor, positive, and negative images
    anchor_input = layers.Input(shape=(28, 28, 1), name='anchor')
    positive_input = layers.Input(shape=(28, 28, 1), name='positive')
    negative_input = layers.Input(shape=(28, 28, 1), name='negative')

    # Generate the embeddings for each input
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    # Return the triplet network
    model = models.Model(inputs=[anchor_input, positive_input, negative_input], 
                         outputs=[anchor_embedding, positive_embedding, negative_embedding])
    return model

6. Create Triplet Generator Function

The next step is to prepare the triplet data. We generate triplets where the anchor and positive come from the same class, and the negative comes from a different class.

def generate_triplets(x_data, y_data):
    """
    This function generates triplets for the triplet loss.
    - anchor: A random sample from a class
    - positive: Another random sample from the same class as anchor
    - negative: A random sample from a different class
    """
    triplets = []
    class_indices = {i: np.where(y_data == i)[0] for i in range(10)}
    
    for _ in range(len(x_data)):
        # Randomly select an anchor class
        anchor_class = random.randint(0, 9)

        # Choose anchor image
        anchor_idx = random.choice(class_indices[anchor_class])
        anchor_image = x_data[anchor_idx]

        # Choose positive image (same class)
        positive_idx = random.choice(class_indices[anchor_class])
        positive_image = x_data[positive_idx]

        # Choose negative image (different class)
        negative_class = random.choice([i for i in range(10) if i != anchor_class])
        negative_idx = random.choice(class_indices[negative_class])
        negative_image = x_data[negative_idx]

        # Append the triplet (anchor, positive, negative)
        triplets.append([anchor_image, positive_image, negative_image])
    
    return np.array(triplets)

# Generate triplets from the training data
triplet_train = generate_triplets(x_train, y_train)
anchor_train, positive_train, negative_train = triplet_train[:, 0], triplet_train[:, 1], triplet_train[:, 2]

7. Compile the Model

Now that we have the triplet network and triplet loss, we can compile the model.

# Create the triplet network
triplet_model = create_triplet_network()

# Compile the model with the triplet loss
triplet_model.compile(optimizer='adam', loss=triplet_loss(margin=1.0))

8. Train the Model

We can now train the model on the MNIST triplets.

# Train the triplet network
history = triplet_model.fit([anchor_train, positive_train, negative_train], 
                            [anchor_train, positive_train, negative_train], 
                            epochs=10, batch_size=32)

9. Evaluating the Model

After training, you can evaluate how well the model performs by checking how close the embeddings are for similar and dissimilar images.

# To check how well the embeddings are performing, you could visualize the embeddings
# For simplicity, let's take a few examples from the test set:

triplet_test = generate_triplets(x_test, y_test)
anchor_test, positive_test, negative_test = triplet_test[:, 0], triplet_test[:, 1], triplet_test[:, 2]

# Extract embeddings for the test set
anchor_embeddings, positive_embeddings, negative_embeddings = triplet_model.predict([anchor_test, positive_test, negative_test])

# Visualize the distance between anchor and positive, anchor and negative
from sklearn.metrics.pairwise import euclidean_distances

# Calculate the Euclidean distances between the embeddings
pos_dist = euclidean_distances(anchor_embeddings, positive_embeddings)
neg_dist = euclidean_distances(anchor_embeddings, negative_embeddings)

# Compare the distances
print("Average distance between anchor and positive: ", np.mean(pos_dist))
print("Average distance between anchor and negative: ", np.mean(neg_dist))

10. Conclusion

    This code defines a Triplet Neural Network (TNN) for the MNIST dataset using Keras/TensorFlow.
    It uses triplet loss to learn embeddings for the MNIST digits.
    The training process involves generating triplets of images (anchor, positive, negative) and learning embeddings that minimize the distance between similar images (anchor and positive) while maximizing the distance to dissimilar images (anchor and negative).

This approach is useful for tasks such as one-shot learning and image similarity, where you want to compare images in an embedding space.
