import tensorflow as tf
import multiprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the number of available CPU cores
num_cpu_cores = multiprocessing.cpu_count()

# Set the maximum number of CPU threads
tf.config.threading.set_inter_op_parallelism_threads(num_cpu_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cpu_cores)

# Set the path to your dataset directory
data_dir = 'Dupli Braces'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
vsl_dir = os.path.join(data_dir, 'valid')

# Image size for ResNet model
image_size = (224, 224)
batch_size = 32

# Create data generators for training, validation, and testing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Set shuffle to False for the test generator
)

validation_generator = datagen.flow_from_directory(
    vsl_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load the pre-trained ResNet50 model without the top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(512, activation='relu')(x)

# Add the final classification layer with softmax activation
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Combine the base model and the new classification layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Make predictions on the validation data
y_true = validation_generator.classes
y_pred_prob = model.predict(validation_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# Compute classification report
report = classification_report(y_true, y_pred, target_names=validation_generator.class_indices, output_dict=True)
print('Classification Report:')
for class_name, metrics in report.items():
    if class_name != 'accuracy':  # Exclude the "accuracy" class
        print(f'Class: {class_name}')
        print(f'Precision: {metrics["precision"]}')
        print(f'Recall: {metrics["recall"]}')
        print(f'F1-score: {metrics["f1-score"]}')
        print(f'Support: {metrics["support"]}')
        print('-------------------')

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=list(validation_generator.class_indices.values()))
print('Confusion matrix:\n', conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=validation_generator.class_indices.keys(),
            yticklabels=validation_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the trained model
model.save('resnet_model_dupli', save_format='tf')

# Save training history for future reference
with open('resnet_training_history_dupli.pkl', 'wb') as file:
    pickle.dump(history.history, file)
