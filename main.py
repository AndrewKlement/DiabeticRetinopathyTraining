import pandas as pd
import tensorflow as tf
from keras import mixed_precision
import keras

keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
mixed_precision.set_global_policy('mixed_float16')

train = pd.read_csv("dataset/train_1.csv")

train_path = train["id_code"].values
train_diag = train["diagnosis"].values

dt_train = tf.data.Dataset.from_tensor_slices((train_path, train_diag))

val = pd.read_csv("dataset/valid.csv")

val_path = val["id_code"].values
val_diag = val["diagnosis"].values

dt_val = tf.data.Dataset.from_tensor_slices((val_path, val_diag))

test = pd.read_csv("dataset/test.csv")

test_path = test["id_code"].values
test_diag = test["diagnosis"].values

dt_test = tf.data.Dataset.from_tensor_slices((test_path, test_diag))

def train_image(path, label):
    base_path = tf.constant("dataset/train_images/train_images/")
    image_path = tf.strings.join([base_path, path, '.png'])
    img = tf.io.read_file(image_path)
    read_img = tf.io.decode_png(img, channels=3, dtype=tf.dtypes.uint8)
    return read_img, label

def val_image(path, label):
    base_path = tf.constant("dataset/val_images/val_images/")
    image_path = tf.strings.join([base_path, path, '.png'])
    img = tf.io.read_file(image_path)
    read_img = tf.io.decode_png(img, channels=3, dtype=tf.dtypes.uint8)
    return read_img, label

def test_image(path, label):
    base_path = tf.constant("dataset/test_images/test_images/")
    image_path = tf.strings.join([base_path, path, '.png'])
    img = tf.io.read_file(image_path)
    read_img = tf.io.decode_png(img, channels=3, dtype=tf.dtypes.uint8)
    return read_img, label

def resize(img, label):
    resized = tf.image.resize(img, [224, 224])
    return resized, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, label

dt_train = dt_train.map(train_image).map(resize).map(augment).batch(5)
dt_val = dt_val.map(val_image).map(resize).batch(5)
dt_test = dt_test.map(test_image).map(resize).batch(1)

model = keras.applications.ResNet50V2(input_shape=(224,224, 3), weights=None, classes=5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

training = model.fit(dt_train, epochs=20, verbose=2, validation_data=dt_val, callbacks=early_stopping)

model.save_weights('./checkpoints/V3.weights.h5')

test_loss, test_acc = model.evaluate(dt_test)
