 
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

