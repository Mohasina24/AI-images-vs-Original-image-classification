from typing import Tuple
import tensorflow as tf

def simple_cnn(input_shape: Tuple[int,int,int]=(128,128,3), dropout=0.3, l2=1e-4):
    reg = tf.keras.regularizers.l2(l2)
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1./255)(inputs)
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    x = data_aug(x)

    for f in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
        x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs, name='SimpleCNN')

def efficientnet_b0(input_shape=(128,128,3), dropout=0.3, freeze_backbone=True):
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    x = data_aug(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='EfficientNetB0_binary')

def vgg16_tl(input_shape=(128,128,3), dropout=0.3, freeze_backbone=True):
    base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = not freeze_backbone
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.RandomFlip('horizontal')(x)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='VGG16_TL')

def resnet50_tl(input_shape=(128,128,3), dropout=0.3, freeze_backbone=True):
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = not freeze_backbone
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.RandomFlip('horizontal')(x)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='ResNet50_TL')
