import tensorflow as tf

def convolution2d(inputs, filters, kernel_size, strides, padding="same", name="conv2d", reuse=False):
    with tf.variable_scope(name):
        outputs = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
                bias_initializer=None,
                reuse=reuse)
        return outputs

def residual_block(inputs, filters, name="residual", reuse=False):
    """Implementation of the residual block.
    Reference: Deep Residual Learning for Image Recognition (He et al. 2015)"""
    with tf.variable_scope(name):
        outputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        outputs = convolution2d(outputs, filters, 3, 1, padding="valid", name=name+"_conv1", reuse=reuse)
        outputs = instance_normalization(outputs, name=name+"_inorm1", reuse=reuse)
        outputs = tf.nn.relu(outputs, name=name+"_relu")
        outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        outputs = convolution2d(outputs, filters, 3, 1, padding="valid", name=name+"_conv2", reuse=reuse)
        outputs = instance_normalization(outputs, name=name+"_inorm2", reuse=reuse)
        outputs = outputs + inputs
        return outputs

def deconvolution2d(inputs, filters, kernel_size, strides, padding="same", name="deconv2d", reuse=False):
    with tf.variable_scope(name):
        outputs = tf.layers.conv2d_transpose(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
                bias_initializer=None,
                reuse=reuse)
        return outputs

def leaky_relu(inputs, alpha=0.2, name="lrelu"):
    with tf.variable_scope(name):
        outputs = tf.maximum(inputs, alpha * inputs, name=name)
    return outputs

def instance_normalization(inputs, name="inorm", reuse=False):
    """Implementation of instance normalization, which is used instead of batch normalization.
    Reference: Instance Normalization: The Missing Ingredient for Fast Stylization (Ulyanov et. al
    2017)"""
    with tf.variable_scope(name, reuse=reuse):
        depth = inputs.get_shape()[3]
        gamma = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.01))
        beta = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(1.0))
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        std_inverse = tf.rsqrt(variance + 0.00001)
        outputs = gamma * (inputs - mean) * std_inverse + beta
        return outputs

def generator(inputs, name="generator", reuse=False):
    """Implementation of the generator architecture. Namely:
    c7s1-32, d64, d128, 9x R128, u64, u32, c7s1-3
    Reference: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu
    et al. 2017)"""
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            tf.variable_scope(scope, reuse=False)
            assert scope.reuse == False
    
    # convolution part
    pad1 = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    conv1 = convolution2d(pad1, 32, 7, 1, padding="valid", name=name+"_c7s1-32_a", reuse=reuse)
    inorm1 = instance_normalization(conv1, name=name+"_c7s1-32_b", reuse=reuse)
    relu1 = tf.nn.relu(inorm1, name=name+"_c7s1-32_c")
    
    conv2 = convolution2d(relu1, 64, 3, 2, name=name+"_d64_a", reuse=reuse)
    inorm2 = instance_normalization(conv2, name=name+"_d64_b", reuse=reuse)
    relu2 = tf.nn.relu(inorm2, name=name+"_d64_c")

    conv3 = convolution2d(relu2, 128, 3, 2, name=name+"_d128_a", reuse=reuse)
    inorm3 = instance_normalization(conv3, name=name+"_d128_b", reuse=reuse)
    relu3 = tf.nn.relu(inorm3, name=name+"_d128_c")

    # residual part
    res1 = residual_block(relu3, 128, name=name+"_r128_1", reuse=reuse)
    res2 = residual_block(res1, 128, name=name+"_r128_2", reuse=reuse)
    res3 = residual_block(res2, 128, name=name+"_r128_3", reuse=reuse)
    res4 = residual_block(res3, 128, name=name+"_r128_4", reuse=reuse)
    res5 = residual_block(res4, 128, name=name+"_r128_5", reuse=reuse)
    res6 = residual_block(res5, 128, name=name+"_r128_6", reuse=reuse)
    res7 = residual_block(res6, 128, name=name+"_r128_7", reuse=reuse)
    res8 = residual_block(res7, 128, name=name+"_r128_8", reuse=reuse)
    res9 = residual_block(res8, 128, name=name+"_r128_9", reuse=reuse)

    # deconvolution part
    dconv1 = deconvolution2d(res9, 64, 3, 2, name=name+"_u64_a", reuse=reuse)
    inorm4 = instance_normalization(dconv1, name=name+"_u64_b", reuse=reuse)
    relu4 = tf.nn.relu(inorm4, name=name+"_u64_c")

    dconv2 = deconvolution2d(relu4, 32, 3, 2, name=name+"_u32_a", reuse=reuse)
    inorm5 = instance_normalization(dconv2, name=name+"_u32_b", reuse=reuse)
    relu5 = tf.nn.relu(inorm5, name=name+"_u64_c")

    # output part
    pad2 = tf.pad(relu5, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    conv4 = convolution2d(pad2, 3, 7, 1, padding="valid", name=name+"_c7s1-3_a", reuse=reuse)
    inorm6 = instance_normalization(conv4, name=name+"_c7s1-3_b", reuse=reuse)
    output = tf.tanh(inorm6, name=name+"c7s1-3_c")

    return output

def discriminator(inputs, use_sigmoid=False, name="discriminator", reuse=False):
    """Implementation of the discriminator architecture. Namely:
    c64, c128, c256, c512
    Reference: * see generator reference
    * Image-to-Image Translation with Conditional Adversarial Networks (Isola et. al 2016)"""

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            tf.variable_scope(scope, reuse=False)
            assert scope.reuse == False
    
    # convolution part
    conv1 = convolution2d(inputs, 64, 4, 2, name=name+"_c64_a", reuse=reuse)
    lrelu1 = leaky_relu(conv1, name=name+"_c64_c")
    
    conv2 = convolution2d(lrelu1, 128, 4, 2, name=name+"_c128_a", reuse=reuse)
    inorm2 = instance_normalization(conv2, name=name+"_c128_b", reuse=reuse)
    lrelu2 = leaky_relu(inorm2, name=name+"_c128_c")

    conv3 = convolution2d(lrelu2, 256, 4, 2, name=name+"_c256_a", reuse=reuse)
    inorm3 = instance_normalization(conv3, name=name+"_c256_b", reuse=reuse)
    lrelu3 = leaky_relu(inorm3, name=name+"_c256_c")

    conv4 = convolution2d(lrelu3, 512, 4, 1, name=name+"_c512a", reuse=reuse)
    inorm4 = instance_normalization(conv4, name=name+"_c512_b", reuse=reuse)
    lrelu4 = leaky_relu(inorm4, name=name+"_c512_c")
    # TO-DO: 
    # * clarify why other implementations use stride of 1 here rather than following paper!
    # Note:
    # In https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py paper
    # authors use indeed stride of 1 for last 2 layers.... *why not write in paper?*
    
    # output part
    conv5 = convolution2d(lrelu4, 1, 4, 1, name=name+"_cout_a", reuse=reuse)
    output = tt.sigmoid(conv5, name=name+"cout_b") if use_sigmoid else conv5

    return output


