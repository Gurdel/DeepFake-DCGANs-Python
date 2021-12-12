# DeepFake-DCGANs-Python

Deepfake image generation using own and third-party GAN

Notebooks with last DCGAN models are gan_last.ipynb (MNIST) and gan_color.ipynb 
(AVHN). Other files named dcgan* are previous version with another GAN models.

## Agenda

>1. [Introduction](#Introduction)
>2. [Build your own DCGAN:](#Build-your-own-DCGAN)
>>- [import modules](#1-import-modules)
>>- [download dataset](#2-download-dataset)
>>- [generator model](#3-generator-model)
>>- [discriminator model](#4-discriminator-model)
>>- [DCGAN](#5-DCGAN)
>>- [train model](#6-train-model)
>3. [Use other datasets](#Use-other-datasets)
>4. [Third-party GANs](#Third-party-GANs)
>>- [StyleGAN](#stylegan)
>>- [Pretrained StyleGAN](#pretrained-stylegan)
>>- [StyleGAN2](#StyleGAN2)
>>- [Pretrained StyleGAN2](#Pretrained-StyleGAN2)
>>- [StyleGAN2 + ADA](#stylegan2--ada)
>>- [Lightweight GAN](#Lightweight-GAN)
>>- [Pretrained GANs with Tensorflow](#pretrained-gans-with-tensorflow)
>5. [Deepfake Websites](#Deepfake-Websites)
>>- [AutoDraw](#autodraw---how-to-make-a-doodle-drawing)
>>- [Quick, Draw!](#quick-draw)
>>- [ThisPersonDoesNotExist](#thispersondoesnotexist---generate-a-person)
>>- [Sematris](#sematris---neurotetris)
>>- [Floom](#Floom)
>>- [cleanup.pictures](#cleanup.pictures)
>>- [ThisCatDoesNotExist](#thiscatdoesnotexist---create-a-cat)
>>- [remove.bg](#removebg---remove-the-background-from-the-photo)
>>- [Teachable Machine](#teachable-machine---independently-train-a-neural-network)
>>- [gnod.com](#gnodcom---finding-a-movie-for-the-evening-is-no-longer-a-problem)
>>- [Artbreeder](#artbreeder---generates-random-faces-abstractions-covers-and-landscapes)
>>- [Dream neural network](#dream-neural-network---generates-pictures-by-text-request)

___
## Introduction

Deep fake (also spelled deepfake) is a type of artificial intelligence used to 
create convincing images, audio and video hoaxes. The term, which describes both 
the technology and the resulting bogus content, is a portmanteau of deep learning 
and fake.

Deepfake content is created by using two competing AI algorithms -- one is called 
the generator and the other is called the discriminator. The generator, which 
creates the phony multimedia content, asks the discriminator to determine whether 
the content is real or artificial.

Together, the generator and discriminator form something called a generative 
adversarial network (GAN). Each time the discriminator accurately identifies 
content as being fabricated, it provides the generator with valuable information 
about how to improve the next deepfake.

The first step in establishing a GAN is to identify the desired output and create 
a training dataset for the generator. Once the generator begins creating an 
acceptable level of output, it can be fed to the discriminator.

As the generator gets better at creating fake images, the discriminator gets 
better at spotting them. Conversely, as the discriminator gets better at 
spotting fake image, the generator gets better at creating them.

A Deep Convolutional GAN (DCGAN) is a direct extension of the GAN described 
above, except that it explicitly uses convolutional and convolutional-transpose 
layers in the discriminator and generator, respectively. The discriminator is 
made up of strided convolution layers, batch norm layers, and LeakyReLU 
activations. The input is a RGB input image and the output is a scalar 
probability that the input is from the real data distribution. The generator 
is comprised of convolutional-transpose layers, batch norm layers, and ReLU 
activations. The input is a latent vector, that is drawn from a standard normal 
distribution and the output is a RGB image. The strided conv-transpose layers 
allow the latent vector to be transformed into a volume with the same shape as an image.

For more detailed information about all layers I used in this work it’s better to 
read official TensorFlow documentation: 
https://www.tensorflow.org/api_docs/python/tf/keras/layers.

___
## Build your own DCGAN

### 1) import modules

```python
import tensorflow as tf
import numpy as np
import random

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Reshape, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from time import time
```
### 2) download dataset

The first and main dataset for my work is MNIST database of handwritten digits,
has a training set of 60,000 examples, and a test set of 10,000 examples. 
It is a subset of a larger set available from NIST. The digits have been 
size-normalized and centered in a fixed-size image. It is a good database 
for people who want to try learning techniques and pattern recognition methods on 
real-world data while spending minimal efforts on preprocessing and formatting.

You can download this dataset using builded-in tfTensorFlow loader and then 
concatenate test and training data for getting more real image examples. Then 
we need to normalize data to range [0, 1], because generator generates values 
in this range.
```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

x = x / 255 
```
Let's plot some examples from dataset:
```python
fig = plt.figure(figsize=(4,4))
imgs = random.choices(x, k=16)
for i in range(len(imgs)):
    plt.subplot(4, 4, i+1)
    plt.imshow(imgs[i])
    plt.axis('off')
plt.show()
```
![img.png](./images/mnist_examples.png)

We will generate only one number for better quality and performance. 
So we need to filter data
```python
indices = np.where(y==3)[0]
x = x[indices]
```
Real data samples after filtering:

![img.png](./images/mnist_3.png)

### 3) generator model

Let's build generator model. Input data is vector of random normal 
distributed values with length 128. There are 5 Conv2DTranspose layers in the 
model. Input shape is (128, ) and output shape is (28, 28, 1), like in real data.
```python
noise_size = 128
generator = Sequential([
    Dense(512, input_shape=[noise_size]),
    Reshape([1,1,512]),
                        
    Conv2DTranspose(512, kernel_size=4, use_bias=False),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(256, kernel_size=4, use_bias=False),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(),

    Conv2DTranspose(64, kernel_size=4, padding='same', use_bias=False),
    LeakyReLU(),
    BatchNormalization(),

    Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation='sigmoid'),
])
```

Total params: 7,017,729

Trainable params: 7,015,809

Non-trainable params: 1,920

You can experiment yourself with the structure of the model, the number of 
layers and their parameters. The main thing is that the output of the generator 
matches the input data of the dataset into the classifier.

### 4) discriminator model

Now we can build the discriminator model. I chose next configuration:
```python
discriminator = Sequential([
    Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=[28,28,1]),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False),

    Flatten(),
    Dense(1, activation='sigmoid'),
])
```
There are 5 Conv2D layers in the model. Input shape is (28, 28, 1), 
equal to mnist image size, and output is one number, and output is only one 
value with activation function sigmoid. It’s used to predict if image is real (1) 
or generated by generator model (0).

Total params: 6,952,257

Trainable params: 6,950,337

Non-trainable params: 1,920

### 5) DCGAN

For building GAN, I just need to place generator output to discriminator input 
and check the accuracy of results. To check the accuracy of the model, I use 
the accuracy metric and the binary crosentropy loss function, 
since we have only two classes: real and fake images.
```python
opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

input_layer = tf.keras.layers.Input(shape=(noise_size, ))
gen_out = generator(input_layer)
disc_out = discriminator(gen_out)

gan = Model(
    input_layer,
    disc_out
)

discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
```
Total params: 13,969,986

Trainable params: 7,015,809

Non-trainable params: 6,954,177

### 6) train model

Now we can train our model. For this example it's enough to train model for 50 epochs, 
but for more complex dataset we need more epochs. Every training batch contains 128 
real and 128 fake images. In each iteration (Niter = Nexp / Nbatch) we save models 
loss and accuracy into variable for plot it after training.
```python
epochs = 50
batch_size = 256
half = int(batch_size/2)
steps_per_epoch = int(2 * x.shape[0]/batch_size)
stat = {}
iter = 0
seed = tf.random.normal([num_examples, noise_size])

for e in range(epochs):
    for step in range(steps_per_epoch):
        true_examples = x[half*step:half*(step+1)]
        true_examples = np.reshape(true_examples, (true_examples.shape[0], 28, 28, 1))
        
        noise = np.random.randn(half, noise_size)
        gen_examples = generator.predict(noise)
        
        x_batch = np.concatenate([gen_examples, true_examples], axis=0)
        y_batch = np.array([0]*half + [1]*half)
        
        indices = np.random.choice(range(batch_size), batch_size, replace=False)
        
        x_batch = x_batch[indices]
        y_batch = y_batch[indices]
        
        discriminator.trainable = True
        discriminator.train_on_batch(x_batch, y_batch)
        discriminator.trainable = False
        
        gan_loss, gan_acc = gan.train_on_batch(noise, np.ones((half, 1)))
        dis_loss, dis_acc = discriminator.evaluate(x_batch, y_batch, verbose=False)
        stat[iter] = [gan_loss, gan_acc, dis_loss, dis_acc]
        iter += 1
```
For plotting generated images you can use next script:
```python
fig = plt.figure(figsize=(5,5))
imgs = generator(seed, training=False)
imgs = np.reshape(imgs, (imgs.shape[0], 28, 28))
for i in range(len(imgs)):
    plt.subplot(5, 5, i+1)
    img = np.array(imgs[i] * 127.5 + 127.5, dtype='uint')
    plt.imshow(img)
    plt.axis('off')
plt.show()
```
I got next results

![img3.png](./images/example-3.png)

In next images you can see how changes models’ accuracy and losses. 
Discriminator accuracy 0.5 means that it can’t recognize fake images 
from real. You can also notice that the indicators are in antiphase: 
when the accuracy of the discriminator increases, the accuracy of the 
generator decreases and vice versa.

![img3.png](./images/loss.png)
![img3.png](./images/acc.png)

___
## Use other datasets

As you can see, for one of MNIST number we got good results. We can 
change number for generating. For example, for zeros I got next 
images after 42 epochs 

![img3.png](./images/example0.png)

But even if we start generating all the numbers from the MNIST dataset, 
the result will not be so good.

![img3.png](./images/example-all.png)

I also tried to use this DCGAN for generating much complex images, 
like number nine from SVHN.

Street View House Number (SVHN) is a digit classification benchmark dataset 
that contains 600000 32×32 RGB images of printed digits (from 0 to 9) cropped 
from pictures of house number plates. The cropped images are centered in the 
digit of interest, but nearby digits and other distractors are kept in the image. 
SVHN has three sets: training, testing sets and an extra set with 530000 images 
that are less difficult and can be used for helping with the training process.

![img3.png](./images/example9.png)

I am not satisfied with results. I got the outlines of the desired numbers, 
but they are very easy to distinguish from the original. Perhaps the model 
just needs more epochs to train, but I don't have enough computing power.

___
## Third-party GANs

As you can see, this DCGAN works poorly for complex examples. Therefore, it is 
better to use ready-made tested more complex solutions. I will list them below. 
I'll leave links to their documentation. It makes no sense to write examples of 
their use myself, since it will just be a copy and paste of the documentation.

### StyleGAN

A Style-Based Generator Architecture for Generative Adversarial Network

StyleGAN — Official TensorFlow Implementation: https://github.com/NVlabs/stylegan

An alternative generator architecture for generative adversarial networks, 
borrowing from style transfer literature. The new architecture leads to an 
automatically learned, unsupervised separation of high-level attributes (e.g., 
pose and identity when trained on human faces) and stochastic variation in the 
generated images (e.g., freckles, hair), and it enables intuitive, scale-specific 
control of the synthesis. The new generator improves the state-of-the-art in terms 
of traditional distribution quality metrics, leads to demonstrably better interpolation 
properties, and also better disentangles the latent factors of variation. To quantify 
interpolation quality and disentanglement, we propose two new, automated methods that are 
applicable to any generator architecture. Finally, we introduce a new, highly 
varied and high-quality dataset of human faces.

### Pretrained StyleGAN

https://github.com/justinpinkney/awesome-pretrained-stylegan

A collection of pre-trained StyleGAN models trained on different 
datasets at different resolution to download.

### StyleGAN2

Analyzing and Improving the Image Quality of StyleGAN

StyleGAN2 — Official TensorFlow Implementation: https://github.com/NVlabs/stylegan2

The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven 
unconditional generative image modeling. We expose and analyze several of its characteristic 
artifacts, and propose changes in both model architecture and training methods to address
them. In particular, we redesign the generator normalization, revisit progressive growing,
and regularize the generator to encourage good conditioning in the mapping from latent codes 
to images. In addition to improving image quality, this path length regularizer yields the 
additional benefit that the generator becomes significantly easier to invert. This makes it
possible to reliably attribute a generated image to a particular network. We furthermore 
visualize how well the generator utilizes its output resolution, and identify a capacity 
problem, motivating us to train larger models for additional quality improvements. Overall, 
our improved model redefines the state of the art in unconditional image modeling, both in 
terms of existing distribution quality metrics as well as perceived image quality.

### Pretrained StyleGAN2

https://github.com/justinpinkney/awesome-pretrained-stylegan2

A collection of pre-trained StyleGAN2 models trained on different datasets at 
different resolution to download.

### StyleGAN2 + ADA

Training Generative Adversarial Networks with Limited Data

StyleGAN2 with adaptive discriminator augmentation (ADA) - 
Official TensorFlow implementation: https://github.com/NVlabs/stylegan2-ada

Training generative adversarial networks (GAN) using too little data typically leads to 
discriminator overfitting, causing training to diverge. We propose an adaptive discriminator 
augmentation mechanism that significantly stabilizes training in limited data regimes. The 
approach does not require changes to loss functions or network architectures, and is 
applicable both when training from scratch and when fine-tuning an existing GAN on another 
dataset. We demonstrate, on several datasets, that good results are now possible using only 
a few thousand training images, often matching StyleGAN2 results with an order of magnitude
fewer images. We expect this to open up new application domains for GANs. We also find that 
the widely used CIFAR-10 is, in fact, a limited data benchmark, and improve the record FID 
from 5.59 to 2.42.

### Lightweight GAN

https://github.com/lucidrains/lightweight-gan

Implementation of 'lightweight' GAN proposed in ICLR 2021, in Pytorch. The main 
contributions of the paper is a skip-layer excitation in the generator, paired
with autoencoding self-supervised learning in the discriminator. Quoting the 
one-line summary "converge on single gpu with few hours' training, on 1024 
resolution sub-hundred images".

### Pretrained GANs with Tensorflow

https://github.com/kozistr/Awesome-GANs

You can download pre-trained models from Google Drive

___
## Deepfake Websites

### AutoDraw - how to make a doodle drawing

https://www.autodraw.com/ 

Artificial Intelligence and Machine Learning are helping to turn sloppy 
sketches into crisp, rendered images. The user only needs to draw a few 
lines on the canvas for the algorithm to suggest the planned drawing.
Artificial intelligence compares images from an extensive database and 
selects the appropriate options.

### Quick, Draw! 

https://quickdraw.withgoogle.com/

A very fun neuro-game. The user creates drawings and 
prompts Google's algorithms to guess what they mean.
The learning model improves as the number of guessed images increases. The
model answers in Russian, sometimes very funny

### ThisPersonDoesNotExist - generate a person

https://thispersondoesnotexist.com/

The neural network creates a realistic image of a human face. 
A new image appears every time you open or refresh the page. 
The algorithm is based on Nvidia's StyleGAN generative neural network.

### Sematris - neurotetris

https://research.google.com/semantris/

Google's minigame works in two ways: Thoughtful Tetris or Intense Arcade.

In the first case, you need to write a related word to one of those presented in 
the list, and the neural network will try to guess which of these words it fits.

In the second, choose a word close in meaning to the variant proposed by the
algorithm. For example, match the word "sleep" with the word "bed". The more
matches, the more points the user gets. Correct word between blocks leads to 
block deletion

### Floom

https://floom.withgoogle.com/

What happens if you drill a hole right under your feet to the other end 
of the planet?

### cleanup.pictures

https://cleanup.pictures/

The neural network from cleanup.pictures will remove unnecessary items from image, 
you just need to draw the required area with the cursor.

### ThisCatDoesNotExist - create a cat

https://thiscatdoesnotexist.com/

The ThisPersonDoesNotExist project has a logical 
continuation.

Artificial intelligence generates an image of a cat based on the knowledge 
it has gained by analyzing real animal images.

To get an image of a cat, just refresh the website page.

### remove.bg - remove the background from the photo

https://www.remove.bg/

The service allows you to remove the background from a photo in five seconds 
without using graphic editors. Using algorithms, the application selects 
objects in the foreground and removes unnecessary.

### Teachable Machine - independently train a neural network

https://teachablemachine.withgoogle.com/

Teach your computer to recognize images, sounds and poses.

Want to understand how a neural network works? Be sure to test this 
service! Google developers have created an application that will help 
you understand how neural networks work. To conduct the experiment, you will 
need a device with a working webcam and an object to which the new neural 
network will respond.

Artificial intelligence will remember your movements and match them 
with a programmed response, responding to gestures with a GIF image, 
sound or speech.

### gnod.com - finding a movie for the evening is no longer a problem!

https://www.gnovies.com/

This artificial intelligence will plan your leisure time. List your 
favorite musicians, writers or artists and Gnod will suggest other 
people's creations that match your tastes. The same works with TV shows, 
films or cartoons.

### Artbreeder - generates random faces, abstractions, covers and landscapes

https://www.artbreeder.com/

The service has a variety of tools, it is divided into the following sections:

1. "General" - is a mixing of several images. For example, mix images 
   of a burger and a dog. Most of the time, the mixing result is pretty creepy.

2. "Portraits" - this is the strongest side of the service. In this section, 
   you can not only mix images with each other, but also change the structure 
   of a person's portrait: physique, age, gender, mood, realistic / drawn facial 
   outlines, and so on. The result is almost always impressive.

3. "Albums" is the generation of abstract images in the manner of music album 
   covers.

4. "Landscapes" - the section works similarly to the others, but here the 
   emphasis is on landscapes. Surreal landscapes come out best here.

### Dream neural network - generates pictures by text request

https://app.wombo.art/

It turns out abstractly, but quickly and almost always beautiful

The service is designed to create unique wallpapers for a smartphone, but in social networks 
it is used for illustrations for books, comics and dissertations.

To generate a picture, it is enough to enter a short description in English, select one of the
styles (or the item "without a specific style") and click the "Create" button. The whole process
takes just a couple of minutes, after which the finished image is displayed on a separate page in a stylized frame.





