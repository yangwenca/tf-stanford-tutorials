import tensorflow as tf

from layers import *


def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    con1_out = conv(input, 'conv1', [3, 3, 1], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu) 
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    con2_out = conv(con1_out, 'conv2', [3, 3, 8], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu) 
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    con3_out = conv(con2_out, 'conv3', [3, 3, 8], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu) 
    # FC: output_dim: 100, no non-linearity
    output = fc(con3_out, 'encoder_fc', 100, non_linear_fn=None)
    return output

def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    fc_out = fc(input, 'decoder_fc', 128, non_linear_fn=tf.nn.relu)
    # Reshape to [batch_size, 4, 4, 8], use [-1,4,4,8]
    fc_reshape = tf.reshape(fc_out, shape=[-1, 4, 4, 8])
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    deconv1_out = deconv(fc_reshape, 'deconv1', [3, 3, 8], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu)
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    deconv2_out = deconv(deconv1_out, 'deconv2', [8, 8, 1], [2, 2], padding='VALID', non_linear_fn=tf.nn.relu)
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    output = deconv(deconv2_out, 'deconv3', [7, 7, 1], [1, 1], padding='VALID', non_linear_fn=tf.nn.sigmoid)
    return output

def autoencoder(input_shape):
    # Define place holder with input shape
    # input_shape is [batch_size, 28, 28, 1]
    input_image = tf.placeholder(tf.float32, input_shape, name="input_placeholder")
    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoder_out = encoder(input_image)
        # Pass encoding into decoder to obtain reconstructed image
        reconstructed_image = decoder(encoder_out)
        # Return input image (placeholder) and reconstructed image
        return input_image, reconstructed_image
