import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

#Skapar dataset och ignorerar errors (fick error för fel filtyp)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "./train",
    image_size=(224, 224),
    batch_size=32
).apply(tf.data.experimental.ignore_errors())

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "./validation",
    image_size=(224, 224),
    batch_size=32
).apply(tf.data.experimental.ignore_errors())

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "./test",
    image_size=(224, 224),
    batch_size=32
).apply(tf.data.experimental.ignore_errors())

#laddar MobileNetV2 och fryser vikterna
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

#bygger modellen
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#träna modellen i 10 epochs
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

#testar modellen med testdata
loss, accuracy = model.evaluate(test_data)
print(f"Test accuracy: {accuracy:.2f}")

#sparar modellen
model.save("mobilenetv2_base.keras")

base_model.trainable = True

#kompilera om modellen med en lägre inlärningshastighet
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#lite finetuning c:
history_finetune = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

#Utvärdera modellen igen på testdata efter fine-tuning
loss, accuracy = model.evaluate(test_data)
print(f"Test accuracy after fine-tuning: {accuracy:.2f}")

# 🚀 Spara den finjusterade modellen
model.save("mobilenetv2_finetuned.keras")

print("woohooooooo")
