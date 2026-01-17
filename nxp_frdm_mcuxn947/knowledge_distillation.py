import tensorflow as tf
import numpy as np
import os
import glob
from tensorflow.keras import layers, models

TEACHER_PATH = "model/person_det_160x128.tflite"
DATA_DIR = "coco_person_mini"
IMG_W, IMG_H = 160, 128
BATCH_SIZE = 32
EPOCHS = 75
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

interpreter = tf.lite.Interpreter(model_path=TEACHER_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
teacher_out_shape = output_details[0]["shape"]


def data_gen(subset):
    file_list = glob.glob(os.path.join(DATA_DIR, "images", subset, "*.jpg"))
    for file_path in file_list:
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_H, IMG_W))

        student_input = tf.cast(img, tf.float32) / 255.0

        teacher_input = tf.cast(tf.image.resize(img, (IMG_H, IMG_W)), tf.int8)
        teacher_input = tf.expand_dims(teacher_input, axis=0)

        interpreter.set_tensor(input_index, teacher_input)
        interpreter.invoke()
        target = interpreter.get_tensor(output_index)

        yield student_input, tf.squeeze(target)


def get_dataset(subset):
    output_sig = (
        tf.TensorSpec(shape=(IMG_H, IMG_W, 3), dtype=tf.float32),
        tf.TensorSpec(shape=teacher_out_shape[1:], dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: data_gen(subset), output_signature=output_sig
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = get_dataset("train")
val_ds = get_dataset("val")


def create_model(scale, output_shape):
    inputs = layers.Input(shape=(IMG_H, IMG_W, 3))
    x = inputs

    filters = [int(8 * scale), int(16 * scale), int(32 * scale)]

    for f in filters:
        x = layers.SeparableConv2D(f, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    bottleneck_dim = int(64 * scale)
    x = layers.Dense(bottleneck_dim, activation="relu")(x)

    flat_dim = np.prod(output_shape[1:])
    x = layers.Dense(flat_dim)(x)
    outputs = layers.Reshape(output_shape[1:])(x)

    return models.Model(inputs, outputs)


def representative_dataset_gen():
    for img, _ in train_ds.take(10):
        for i in range(img.shape[0]):
            yield [tf.expand_dims(img[i], 0)]


configs = [("Tiny", 0.5), ("Medium", 1.0), ("Large", 1.5)]

for name, scale in configs:
    print(f"--- Processing Student: {name} (Scale {scale}) ---")

    student = create_model(scale, teacher_out_shape)
    print(f"Params: {student.count_params()}")

    student.compile(optimizer="adam", loss="mse")
    student.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    h5_path = os.path.join(MODEL_DIR, f"student_{name.lower()}.h5")
    student.save(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(student)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = os.path.join(MODEL_DIR, f"student_{name.lower()}_quant.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Saved: {tflite_path} ({size_kb:.1f} KB)")

print("--- Distillation Complete ---")
