# Import necessary libraries
import streamlit as st

# Importing external libraries with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
    from tensorflow.keras.datasets import mnist
    import numpy as np
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    st.error(f"Required module not found: {e.name}. Please install it using `pip install {e.name}`.")
    st.stop()

# Streamlit app title
st.title("MNIST Digit Classification with CNN")

# Load the MNIST dataset and build the model
@st.cache_resource  # Cache the model to avoid reloading on each interaction
def load_data_and_model():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
    return model, x_test, y_test

# Load the trained model and dataset
model, x_test, y_test = load_data_and_model()

# Sidebar user input
st.sidebar.header("Input Parameters")
num_images = st.sidebar.slider("Number of test images to display:", min_value=1, max_value=20, value=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
st.write(f"### Test Accuracy: {test_accuracy:.2f}")

# Predict on test data
predictions = model.predict(x_test)

# Display predictions
st.write(f"### Displaying {num_images} test images and their predictions:")
fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
for i in range(num_images):
    axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"Predicted: {np.argmax(predictions[i])}")
    axes[i].axis('off')
st.pyplot(fig)
