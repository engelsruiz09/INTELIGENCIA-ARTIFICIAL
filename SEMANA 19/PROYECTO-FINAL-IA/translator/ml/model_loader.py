import tensorflow as tf, functools, pathlib
MODEL_PATH = pathlib.Path(__file__).with_name("best_model_fold4.keras")

@functools.lru_cache(maxsize=1)
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)
