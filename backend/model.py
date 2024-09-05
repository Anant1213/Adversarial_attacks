import tensorflow as tf

# Load model (can be replaced with any other model)
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model
