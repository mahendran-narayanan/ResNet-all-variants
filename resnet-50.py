import tensorflow as tf
from tensorflow.keras import datasets, layers, models
def ResNet50():

    img_input = layers.Input(shape=(224,224,3))
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(
        img_input
    )
    x = layers.Conv2D(64, 3, strides=1, name="conv1_conv")(x)
    
    x = stack(x)


    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    
    x = layers.Dense(
        10, activation='sigmoid', name="predictions"
    )(x)

    # Create model.
    model = tf.keras.Model(img_input, x)#, name=model_name)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True,name=None):
    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + "_0_convsh"
        )(x)
        shortcut = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + "_0_bnsh"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters, kernel_size, padding="SAME", name=name + "_2_conv"
    )(x)
    x = layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D( 4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_3_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name="L"):
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x

def stack(x):
    x = stack1(x, 64, 3, stride1=1, name="conv2")
    x = stack1(x, 128, 4, name="conv3")
    x = stack1(x, 256, 6, name="conv4")
    return stack1(x, 512, 3, name="conv5")

model = ResNet50()
model.build(input_shape=(None,224,224,3))
model.summary()