import tensorflow as tf
import keras
from keras.utils.data_utils import get_file
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
])

nb_class = 5

vggface_resnet = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
print("Кількість шарів: ", len(vggface_resnet.layers))
vggface_resnet.trainable = False
last_layer = vggface_resnet.get_layer('avg_pool').output
train_dataset = keras.utils.image_dataset_from_directory("faces_base", shuffle=True, batch_size=8, image_size=(224, 224))
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = vggface_resnet(x)
x = Flatten(name='flatten')(x)
out = Dense(nb_class, name='classifier')(x)
custom_vgg_model = keras.Model(inputs, out)
print(custom_vgg_model.summary())
base_learning_rate = 0.0001
custom_vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
history = custom_vgg_model.fit(train_dataset, epochs=500)
prob_model = keras.Sequential([custom_vgg_model, tf.keras.layers.Softmax()])
print(prob_model.summary())
prob_model.save('CustomVggModel.h5')

