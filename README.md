### Goal: 

This project is review of a paper published in August 2015 titled A Neural Algorithm of Artistic Style. It showed how a convolutional neural network (CNN) can be used to "paint" a picture that combines the "content" of one image with the "style" of another.

### Methods and Tools:

VGG19 is a pre-trained deep convolutional neural network. It has been trained on the ImageNet dataset: 14-million images from 1,000 categories. VGG19's primary purpose is to identify objects in images.
VGG19 has lot of layers.  The final layer outputs the probabilities of each of the 1,000 output categories. We used layers up to convolution layer 5_3 for our project.

#### Extracting content

To extract the content from an image, we used convolution layer 5_2. 
Let's say we have a content image (p) and we have a target "blank canvas" (x) onto which we want to extract and "paint" the content. The process is to iteratively tweak the target image , that is initially random, until layer 5_2 outputs are similar to the content image. 

Here's the process in a bit more detail:

Run the content image p through the model to get the activation output of convolution layer 5_2. Let's term that output P5_2. Run the (initially random) target image x through the model to get the output of the same layer 5_2. Let's term that output F5_2.
Calculate the error ("content cost") between the two outputs. We can use simple squared error cost function: ∑(P5_2−F5_2)2. Tweak the target image a little to reduce the error. We back-propagate the error through the model back to the target image, and use that as the basis of our tweaking. Repeat steps 2-4 until satisfied. A process of gradient descent The following lines of code are the essential ones for these steps:

sess.run(net['input'].assign(img_content)) p = sess.run(net[layer_content]) x = net[layer_content] [...] tf.reduce_sum(tf.pow((x - p), 2)) 

#### Extracting style from the picture

The process of extracting the style from an image is very similar. There are two differences:
Instead of using just one convolution layer we used a blend of layers 1_1, 2_1, 3_1, 4_1 and 5_1. All five layers contribute equally to the total style cost. Before we pass the outputs of each layer into the squared error cost function, we first applied a function called the Gram matrix.

#### The Gram matrix:

First, understand the purpose of the Gram matrix. Gatys et al. say, "We built a style representation that computes the correlations between the different filter responses." They key word here is correlations. Here's the intuition: the Gram matrix aggregates information on similarities across the image and this makes it blind to local, specific objects. In this fashion, it captures stylistic aspects of the image.

Now the mathematics. Start by recalling that a convolution layer applies a set of filters over the image, and the output of the layer will itself be a three-dimensional matrix. Let's notate the activation output of a layer as Fij, where j is the position (pixel) and i is the filter. The formula for the Gram matrix is then ∑kFikFjk.
