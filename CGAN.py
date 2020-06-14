#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install tensorflow==1.0.0')


# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[ ]:



get_ipython().system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz')
get_ipython().system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz')
get_ipython().system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz')
get_ipython().system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz')


# In[ ]:


get_ipython().system('mkdir MNIST_Fashion')
get_ipython().system('cp *.gz MNIST_Fashion/')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_Fashion/", one_hot = True)


# In[ ]:


print(mnist.train.images.shape)
print(mnist.test.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.labels.shape)


# In[ ]:


#Training PArams
learning_rate = 0.0002
batch_size = 128
epochs = 100000

#Network params
image_dim = 784 #img sz is 28x28
Y_dimension = 10 # The number of classes
gen_hidd_dim = 256
disc_hidd_dim  = 256
z_noise_dim = 100

def xavier_init(shape):
    return tf.random_normal(shape = shape, stddev= 1./tf.sqrt(shape[0]/2.0))


# In[ ]:


weights = {
    "disc_H" : tf.Variable(xavier_init([image_dim + Y_dimension, disc_hidd_dim])),
    "disc_final": tf.Variable(xavier_init([disc_hidd_dim,1])),
    "gen_H": tf.Variable(xavier_init([z_noise_dim + Y_dimension, gen_hidd_dim])),
    "gen_final": tf.Variable(xavier_init([gen_hidd_dim, image_dim]))
}

bias = {
    "disc_H" : tf.Variable(xavier_init([disc_hidd_dim])),
    "disc_final": tf.Variable(xavier_init([1])),
    "gen_H": tf.Variable(xavier_init([gen_hidd_dim])),
    "gen_final": tf.Variable(xavier_init([image_dim]))
}


# In[ ]:



#define placeholders for external input

z_input = tf.placeholder(tf.float32, shape = [None, z_noise_dim], name = "input_noise")
x_input = tf.placeholder(tf.float32, shape = [None, image_dim], name = "real_input")
Y_input = tf.placeholder(tf.float32, shape = [None, Y_dimension], name = "Labels")


# In[ ]:


def Discriminator(x,y):
    inputs = tf.concat(axis = 1, values = [x,y])
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(
        inputs, weights["disc_H"]), bias["disc_H"]))
    final_layer = (tf.add(tf.matmul(
        hidden_layer, weights["disc_final"]), bias["disc_final"]))
    disc_output = tf.nn.sigmoid(final_layer)
    return final_layer, disc_output


# In[ ]:


#Generator NW
def Generator(x,y):
    inputs = tf.concat(axis = 1, values = [x,y])
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(
        inputs, weights["gen_H"]), bias["gen_H"]))
    final_layer = (tf.add(tf.matmul(
        hidden_layer, weights["gen_final"]), bias["gen_final"]))
    gen_output = tf.nn.sigmoid(final_layer)
    return gen_output


# In[ ]:


# building the GEN NW
output_Gen = Generator(z_input, Y_input) #G(z/y)
 
 # Building the Disc NW
real_output1_Disc, real_output_disc = Discriminator(
    x_input, Y_input)                          # implements D(x/y)
fake_output1_Disc, fake_output_disc = Discriminator(
    output_Gen, Y_input)                       # implements D(G(x/y))


# In[ ]:


# building the GEN NW
output_Gen = Generator(z_input, Y_input) #G(z/y)
 
 # Building the Disc NW
real_output1_Disc, real_output_disc = Discriminator(
    x_input, Y_input)                          # implements D(x/y)
fake_output1_Disc, fake_output_disc = Discriminator(
    output_Gen, Y_input)                       # implements D(G(x/y))


# In[ ]:



#first kind of loss
with tf.name_scope("Discriminator_Loss") as scope:
    Discriminator_Loss = -tf.reduce_mean(tf.log(
        real_output_disc+ 0.0001)+tf.log(1.- fake_output_disc+0.0001))
  
with tf.name_scope("Genetator_Loss") as scope:
    Generator_Loss = -tf.reduce_mean(tf.log(
        fake_output_disc+ 0.0001)) # due to max log(D(G(x)))

  # T-board summary
    Disc_loss_total = tf.summary.scalar("Disc_Total_loss", Discriminator_Loss)
    Gen_loss_total = tf.summary.scalar("Gen_loss", Generator_Loss)


# In[ ]:


# Define the variables

Generator_var = [weights["gen_H"], weights["gen_final"], 
                 bias["gen_H"], bias["gen_final"]]
Discriminator_var = [weights["disc_H"], weights["disc_final"], 
                     bias["disc_H"], bias["disc_final"]]

#Define the optimizer
with tf.name_scope("Optimizer_Discriminator") as scope:
    Discriminator_optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).
    minimize(Discriminator_Loss, var_list = Discriminator_var)

with tf.name_scope("Optimizer_Generator") as scope:
    Generator_optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).
    minimize(Generator_Loss, var_list = Generator_var)


# In[ ]:


# Initialize the variables

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
writer = tf.summary.FileWriter("./log", sess.graph)
epochs=20000
for epoch in range(epochs):
    x_batch, Y_label = mnist.train.next_batch(batch_size)
  
  #Generate noise to feed Discriminator
    z_noise = np.random.uniform(-1.,1.,size = [batch_size, z_noise_dim])
    _, Disc_loss_epoch = sess.run([Discriminator_optimize,Discriminator_Loss],
                                  feed_dict={x_input:x_batch, Y_input:Y_label, z_input:z_noise})
    _, Gen_loss_epoch = sess.run([Generator_optimize,Generator_Loss],
                                 feed_dict={z_input:z_noise, Y_input:Y_label})  
  
  #Running the Discriminator summary
    summary_Disc_loss = sess.run(Disc_loss_total, 
                                 feed_dict = {x_input:x_batch, z_input:z_noise, Y_input:Y_label})
  # Adding the Discriminator summary
    writer.add_summary(summary_Disc_loss, epoch)
  
  #Running the Generator summary
    summary_Gen_loss = sess.run(Gen_loss_total, 
                                feed_dict = {z_input:z_noise, Y_input:Y_label})
  # Adding the Generator summary
    writer.add_summary(summary_Gen_loss, epoch)
  
    if epoch % 100000 == 0:
        print("Steps: {0}: Generator Loss: {1}, 
              Discriminator Loss:{2}".format(epoch, Gen_loss_epoch, Disc_loss_epoch))


# In[ ]:


def generate_plot(samples):
    fig = plt.figure(figsize = (4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace = 0.05, hspace = 0.05)
  
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap = 'gray')
    return fig


# In[ ]:


def create(inp):
    feature_map = { "t-shirt":0,
                 "trouser":1,
                 "pullover":2,
                 "dress":3,
                 "coat":4,
                 "sandal":5,
                 "sirt":6,
                 "sneaker":7,
                 "bag":8,
                 "ankle boot": 9
                }
    samples = 16
    z_noise = np.random.uniform(-1.,1.,size = [samples, z_noise_dim])
    
   #one hot encoding
    Y_label = np.zeros(shape = [samples, Y_dimension])
    Y_label[:, feature_map[inp]] = 1
  
   # run the traineg generator excluding Discriminator
    generated_samples = sess.run(output_Gen, feed_dict = {z_input:z_noise, Y_input:Y_label})
    #plot images
  
   generate_plot(generated_samples)


# In[1]:


create('sandal')


# In[ ]:




