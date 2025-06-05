import tensorflow as tf, pathlib
MODEL_PATH = pathlib.Path(__file__).with_name("best_model_fold4.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print(model.input_shape)