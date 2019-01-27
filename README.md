 
# Artistic Style Transfer using Neural Networks

An extraordinary paper was published in August 2015 titled A Neural Algorithm of Artistic Style. 
It showed how a convolutional neural network (CNN) can be used to "paint" a picture that 
combines the "content" of one image with the "style" of another.

# Using a pre-trained model

To start, we're going to need a CNN. We could build our own, but it's much easier to use something off-the-shelf. Gatys et al. used the pre-trained VGG19 model, and we will too.

VGG19 is a deep convolutional neural network built at the University of Oxford (see the paper: V
y Deep Convolutional Networks for Large-Scale Image Recognition). 

It has been trained on the ImageNet dataset:
14-million images from 1,000 categories. VGG19's primary purpose is to identify objects in images.

"Deep" because there are lots and lots of layers. The black layers are convolution layers, and they appear in blocks. 
By convention, I'll be referring to them using their block - so layer conv2_3 refers to the 
third convolution layer in the second block.

The final layer outputs the probabilities of each of the 1,000 output categories.  
For our purposes, we're only interested in the layers up to convolution layer 5_3.

# Extracting style
To extract the content from an image, Gatys et al. use convolution layer 5_2. It's trivial to feed 
an image into the model and generate output from layer 5_2, but how do we take that output and transfer it onto a new image?

Let's say we have a content image (p) and we have a target "blank canvas" (x) onto which we want to extract and "paint" 
the content. The basic process is to iteratively tweak the (initially random) target 
image until it has layer 5_2 outputs similar to the content image. Here's the process in a bit more detail:

Run the content image p through the model to get the activation output of convolution layer 5_2. Let's term that output P5_2.
Run the (initially random) target image x through the model to get the output of the same layer 5_2. Let's term that output F5_2
Calculate the error ("content cost") between the two outputs. We can use simple squared error cost function: ∑(P5_2−F5_2)2.
Tweak the target image a little to reduce the error. We back-propagate the error through the model back to the target image, and use that as the basis of our tweaking.
Repeat steps 2-4 until satisfied. A process of gradient descent
The following lines of code are the essential ones for these steps:

sess.run(net['input'].assign(img_content))
p = sess.run(net[layer_content])
x = net[layer_content]
  [...]
tf.reduce_sum(tf.pow((x - p), 2))
I think this is pretty straightforward, and we already saw in the last post how 
CNN filters can abstract features like outlines and borders from an image. So I won't do it here but, 
if you want to, it's quite easy to do by tweaking the implementation code below and removing the style loss.

# Extracting style from the picture

The process of extracting the style from an image is very similar. There are two differences:

Instead of using just one convolution layer we are going to use five. Gatys et al. use a blend of layers
1_1, 2_1, 3_1, 4_1 and 5_1. All five layers contribute equally to the total style cost 
(although it is possible to weight them and create different effects).
Before we pass the outputs of each layer into the squared error cost function, 
we first apply a function called the Gram matrix. The Gram matrix looks very simple but it's clever and very subtle.

### The Gram matrix: 

First, understand the purpose of the Gram matrix. Gatys et al. say, "We built a style representation that computes
the correlations between the different filter responses." They key word here is correlations. Here's the intuition: 
the Gram matrix aggregates information on similarities across the image and this makes it blind to local, specific objects.
In this fashion, it captures stylistic aspects of the image.

Now the mathematics. Start by recalling (from last post) that a convolution layer applies a set of filters over the image,
and the output of the layer will itself be a three-dimensional matrix. Let's notate the activation output of a layer as Fij,
where j is the position (pixel) and i is the filter. The formula for the Gram matrix is then ∑kFikFjk. 

Code:

import os
import numpy as np
import scipy.misc
import scipy.io
import math
import tensorflow as tf
from sys import stderr
from functools import reduce
import time  

## Inputs 
file_content_image = 'starry_night_mini.jpg' 
file_style_image = 'klimt_2.jpg'   

## Parameters 
input_noise = 0.1     # proportion noise to apply to content image
weight_style = 2e2 

## Layers
layer_content = 'conv4_2' 
layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers_style_weights = [0.2,0.2,0.2,0.2,0.2]

## VGG19 model
path_VGG19 = 'imagenet-vgg-verydeep-19.mat'
# VGG19 mean for standardisation (RGB)
VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

## Reporting & writing checkpoint images
# NB. the total # of iterations run will be n_checkpoints * n_iterations_checkpoint
n_checkpoints = 10            # number of checkpoints
n_iterations_checkpoint = 10   # learning iterations per checkpoint
path_output = 'output'  # directory to write checkpoint images into


### Helper functions
def imread(path):
    return scipy.misc.imread(path).astype(np.float)   # returns RGB format

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
    
def imgpreprocess(image):
    image = image[np.newaxis,:,:,:]
    return image - VGG19_mean

def imgunprocess(image):
    temp = image + VGG19_mean
    return temp[0] 

# function to convert 2D greyscale to 3D RGB
def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret
 

### Preprocessing
# create output directory
if not os.path.exists(path_output):
    os.mkdir(path_output)

# read in images
img_content = imread(file_content_image) 
img_style = imread(file_style_image) 

# convert if greyscale
if len(img_content.shape)==2:
    img_content = to_rgb(img_content)

if len(img_style.shape)==2:
    img_style = to_rgb(img_style)

# resize style image to match content
img_style = scipy.misc.imresize(img_style, img_content.shape)

# apply noise to create initial "canvas" 
noise = np.random.uniform(
        img_content.mean()-img_content.std(), img_content.mean()+img_content.std(),
        (img_content.shape)).astype('float32')
img_initial = noise * input_noise + img_content * (1 - input_noise)

# preprocess each
img_content = imgpreprocess(img_content)
img_style = imgpreprocess(img_style)
img_initial = imgpreprocess(img_initial)
  

#### BUILD VGG19 MODEL
## with thanks to http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style

VGG19 = scipy.io.loadmat(path_VGG19)
VGG19_layers = VGG19['layers'][0]

# help functions
def _conv2d_relu(prev_layer, n_layer, layer_name):
    # get weights for this layer:
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    # create a conv2d layer
    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b    
    # add a ReLU function and return
    return tf.nn.relu(conv2d)

def _avgpool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Setup network
with tf.Session() as sess:
    a, h, w, d     = img_content.shape
    net = {}
    net['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
    net['conv1_1']  = _conv2d_relu(net['input'], 0, 'conv1_1')
    net['conv1_2']  = _conv2d_relu(net['conv1_1'], 2, 'conv1_2')
    net['avgpool1'] = _avgpool(net['conv1_2'])
    net['conv2_1']  = _conv2d_relu(net['avgpool1'], 5, 'conv2_1')
    net['conv2_2']  = _conv2d_relu(net['conv2_1'], 7, 'conv2_2')
    net['avgpool2'] = _avgpool(net['conv2_2'])
    net['conv3_1']  = _conv2d_relu(net['avgpool2'], 10, 'conv3_1')
    net['conv3_2']  = _conv2d_relu(net['conv3_1'], 12, 'conv3_2')
    net['conv3_3']  = _conv2d_relu(net['conv3_2'], 14, 'conv3_3')
    net['conv3_4']  = _conv2d_relu(net['conv3_3'], 16, 'conv3_4')
    net['avgpool3'] = _avgpool(net['conv3_4'])
    net['conv4_1']  = _conv2d_relu(net['avgpool3'], 19, 'conv4_1')
    net['conv4_2']  = _conv2d_relu(net['conv4_1'], 21, 'conv4_2')     
    net['conv4_3']  = _conv2d_relu(net['conv4_2'], 23, 'conv4_3')
    net['conv4_4']  = _conv2d_relu(net['conv4_3'], 25, 'conv4_4')
    net['avgpool4'] = _avgpool(net['conv4_4'])
    net['conv5_1']  = _conv2d_relu(net['avgpool4'], 28, 'conv5_1')
    net['conv5_2']  = _conv2d_relu(net['conv5_1'], 30, 'conv5_2')
    net['conv5_3']  = _conv2d_relu(net['conv5_2'], 32, 'conv5_3')
    net['conv5_4']  = _conv2d_relu(net['conv5_3'], 34, 'conv5_4')
    net['avgpool5'] = _avgpool(net['conv5_4'])


### CONTENT LOSS: FUNCTION TO CALCULATE AND INSTANTIATION
# with thanks to https://github.com/cysmith/neural-style-tf

# Recode to be simpler: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
def content_layer_loss(p, x):
    _, h, w, d = [i.value for i in p.get_shape()]    # d: number of filters; h,w : height, width
    M = h * w 
    N = d 
    K = 1. / (2. * N**0.5 * M**0.5)
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

with tf.Session() as sess:
    sess.run(net['input'].assign(img_content))
    p = sess.run(net[layer_content])  # Get activation output for content layer
    x = net[layer_content]
    p = tf.convert_to_tensor(p)
    content_loss = content_layer_loss(p, x) 


### STYLE LOSS: FUNCTION TO CALCULATE AND INSTANTIATION

def style_layer_loss(a, x):
    _, h, w, d = [i.value for i in a.get_shape()]
    M = h * w 
    N = d 
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(x, M, N):
    F = tf.reshape(x, (M, N))                   
    G = tf.matmul(tf.transpose(F), F)
    return G

with tf.Session() as sess:
    sess.run(net['input'].assign(img_style))
    style_loss = 0.
    # style loss is calculated for each style layer and summed
    for layer, weight in zip(layers_style, layers_style_weights):
        a = sess.run(net[layer])
        x = net[layer]
        a = tf.convert_to_tensor(a)
        style_loss += style_layer_loss(a, x)
        
### Define loss function and minimise
with tf.Session() as sess:
    # loss function
    L_total  = content_loss + weight_style * style_loss 
    
    # instantiate optimiser
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      L_total, method='L-BFGS-B',
      options={'maxiter': n_iterations_checkpoint})
    
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    sess.run(net['input'].assign(img_initial))
    for i in range(1,n_checkpoints+1):
        # run optimisation
        optimizer.minimize(sess)
        
        ## print costs
        stderr.write('Iteration %d/%d\n' % (i*n_iterations_checkpoint, n_checkpoints*n_iterations_checkpoint))
        stderr.write('  content loss: %g\n' % sess.run(content_loss))
        stderr.write('    style loss: %g\n' % sess.run(weight_style * style_loss))
        stderr.write('    total loss: %g\n' % sess.run(L_total))

        ## write image
        img_output = sess.run(net['input'])
        img_output = imgunprocess(img_output)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        output_file = path_output+'/'+timestr+'_'+'%s.jpg' % (i*n_iterations_checkpoint)
        imsave(output_file, img_output)

