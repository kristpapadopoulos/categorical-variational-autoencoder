#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras, Tensorflow Probability and Eager Execution Implementation of
Categorical Variational Autoencoder

Straight Through Gumbel-Softmax Estimator used as per paper (No Temperature
or Learning Rate Annealing and Hard Prior used)
https://arxiv.org/abs/1611.01144

Code developed from:
    
1) Eric Jang: 
https://github.com/ericjang/gumbel-softmax/blob/master/gumbel_softmax_vae_v2.ipynb

2) Google Seedbank Convolutional Variational Autoencoder
https://tools.google.com/seedbank/seed/5719238044024832

cat_vae_v0.1.py - Sep 24, 2018

Tensorflow 1.10.0
Numpy 1.14.5

@author: Krist Papadopoulos
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp 
tfe = tf.contrib.eager
tf.enable_eager_execution()

# define Tensorflow probability distributions
Bernoulli = tfp.distributions.Bernoulli
OneHotCategorical = tfp.distributions.OneHotCategorical
RelaxedOneHotCategorical = tfp.distributions.RelaxedOneHotCategorical
KL = tfp.distributions.kl_divergence

# define models for inference and generating over number of categorical 
# distributions with latent dimension
class CatVAE(tf.keras.Model):
    def __init__(self, latent_dim, num_dist):
        super(CatVAE, self).__init__()
        
        # latent dimensions
        self.latent_dim = latent_dim
        
        # number of distributions to sample from in latent space
        self.num_dist = num_dist
        
        # inference model to estimate the posterior p(z|x)
        self.inference_net = tf.keras.Sequential(
          [
          tf.keras.layers.InputLayer(input_shape=(784,)),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          tf.keras.layers.Dense(256, activation=tf.nn.relu),
          # no activation for final layer
          tf.keras.layers.Dense(self.latent_dim*self.num_dist),
          tf.keras.layers.Reshape((-1, self.num_dist, self.latent_dim))])
        
        # generative model to estimate the likelihood p(x|z) for samples
        self.generative_net = tf.keras.Sequential(
          [
          tf.keras.layers.InputLayer(input_shape=(self.num_dist, self.latent_dim,)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(256, activation=tf.nn.relu),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          # no activation for final layer
          tf.keras.layers.Dense(784)])

    def encode(self, x):
        logits_z = self.inference_net(x)
        return logits_z

    def reparameterize(self, tau, logits_z):
        # generate latent sample using Gumbel-Softmax for categorical variables
        z = RelaxedOneHotCategorical(tau,logits_z).sample()
        z_hard = tf.cast(tf.one_hot(tf.argmax(z,-1), self.latent_dim), z.dtype)
        z = tf.stop_gradient(z_hard - z) + z
        return z

    def decode(self, z):
        logits_x = self.generative_net(z)
        return logits_x

# define categorical vae loss function with hard prior
def CatVAE_loss(model, x, tau):
    logits_z = model.encode(x)
    z = model.reparameterize(tau, logits_z)
    logits_x = model.decode(z)
    reconstruction_error = tf.reduce_sum(Bernoulli(logits=logits_x).log_prob(x),1)
    logits_pz = tf.ones_like(logits_z) * (1./model.latent_dim)
    q_cat_z = OneHotCategorical(logits=logits_z)
    p_cat_z = OneHotCategorical(logits=logits_pz)
    KL_qp = KL(q_cat_z, p_cat_z)
    KL_qp_sum = tf.reduce_sum(KL_qp,[1,2])
    ELBO = tf.reduce_mean(reconstruction_error - KL_qp_sum)
    loss = -ELBO
    return loss

# compute gradients and loss using Tensorflow Eager execution
def compute_gradients(model, x, tau):
    with tf.GradientTape() as tape:
        loss = CatVAE_loss(model, x, tau)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables, global_step=None):
    return optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

if __name__ == "__main__":
    
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    
    # download mnist dataset and reshape tp (batch #, 728 pixels)
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.
    train_images = train_images.reshape((len(train_images), np.prod(train_images.shape[1:])))
    test_images = test_images.reshape((len(test_images), np.prod(test_images.shape[1:])))
    
    # binarize mnist pixels to 1 or 0
    train_images[train_images >=0.5] = 1
    train_images[train_images < 0.5] = 0

    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000

    # use Tensorflow Dataset API for eager execution
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
    
    # define constants for epochs, # of latent dimensions, # of distributions
    # to sample from in latent space
    # current implementation does not incorporate annealing temperature (tau) 
    # or learning rate
    
    epochs = 10
    latent_dim = 10
    num_dist = 30
    tau0 = 1
    lr_init = 0.001
    
    tau = tfe.Variable(tau0, name = "temperature", trainable = False)
    lr = tfe.Variable(lr_init, name = "learning_rate", trainable=False)
    
    optimizer = tf.train.AdamOptimizer(lr)

    model = CatVAE(latent_dim, num_dist)
    
    train_loss_results = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss_avg = tfe.metrics.Mean()
        start_time = time.time()
        
        for train_x in train_dataset:

            gradients, loss = compute_gradients(model, train_x, tau)
            apply_gradients(optimizer, gradients, model.trainable_variables)
            epoch_loss_avg(loss) 
        
        train_loss_results.append(epoch_loss_avg.result())
        epoch_time = time.time() - start_time
        
        if epoch % 1 == 0:
            print('Epoch: {}, Loss: {}, Time: {}'.format(epoch, 
                  epoch_loss_avg.result(), epoch_time))
            
    # take categorical samples from prior (test_z) to generate images
    M = 100*num_dist
    
    test_z = np.zeros((M,latent_dim))
    
    test_z[range(M),np.random.choice(latent_dim,M)] = 1
    
    test_z = np.reshape(test_z,[100, num_dist, latent_dim])
    
    test_z = test_z.astype('float32')
    
    # generate sample images from prior (test_z)
    predictions = Bernoulli(logits=model.decode(test_z)).mean()
    
    predictions = np.array(predictions).reshape((10,10,28,28))
    
    # split into 10 (1,10,28,28) images, concat along columns -> 1,10,28,280
    predictions = np.concatenate(np.split(predictions,10,axis=0),axis=3)
    
    # split into 10 (1,1,28,280) images, concat along rows -> 1,1,280,280
    predictions = np.concatenate(np.split(predictions,10,axis=1),axis=2)
    
    x_img = np.squeeze(predictions)
    
    plt.figure(figsize=(15,15))
    
    plt.imshow(x_img, cmap='gray', interpolation='none')
    
    plt.title('Generated MNIST Images from 100 Test Samples with CatVAE')
    
    plt.savefig('/Users/KP/Desktop/MNIST_cat_vae_v0.1_sample.png', bbox_inches='tight')
    
    
    
    

        
        


    
    
    