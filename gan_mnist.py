##reference - https://github.com/udacity/deep-learning/tree/master/gan_mnist

#prerequisities
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Dense, Activation, LeakyReLU, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#loading mnist datasets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#generator - input are randonly generated numbers
def make_latent_samples(n_samples, sample_size):
    #return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

#generator  - single hidden layered NN that takes randomly generated numbers as the input and produces 784 datapoints as the image output
generator = Sequential([Dense(128, input_shape = (100,)), LeakyReLU(alpha = 0.01), Dense(784), Activation('tanh')], name = 'generator')

#discriminator - single hidden layered NN that takes in the 784 datapoints as the input from the generator and the MNIST data and outputs the classification as the original or fake image
discriminator = Sequential([Dense(128, input_shape=(784,)), LeakyReLU(alpha = 0.01), Dense(1), Activation('sigmoid')], name = 'discriminator')

#the simplified GAN would be a generator and the discriminator
# It takes the latent sample, and the generator inside GAN produces a digit image which the discriminator inside GAN classifies as real or fake

gan = Sequential([generator, discriminator])

#when training the GAN we only train the generator and not the discriminator, and keep the discriminator weights constant
#we train the discriminator seperately

def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

#preprocessing mnist datasets
def preprocess(x):
    x = x.reshape(-1, 784) # 784=28*28
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    return x

X_train_real = preprocess(X_train)
X_test_real  = preprocess(X_test)

#labels for discriminator - true or fake images
def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])

# hyperparameters
sample_size     = 100     # latent sample size (i.e., 100 random numbers)
g_hidden_size   = 128
d_hidden_size   = 128
leaky_alpha     = 0.01
g_learning_rate = 0.0001  # learning rate for the generator
d_learning_rate = 0.001   # learning rate for the discriminator
epochs          = 100
batch_size      = 64      # train batch size
eval_size       = 16      # evaluate size

#training
y_train_real, y_train_fake = make_labels(batch_size)
y_eval_real,  y_eval_fake  = make_labels(eval_size)

for e in range(epochs):
  for i in range(len(X_train_real)//batch_size):

    X_batch_real = X_train_real[i*batch_size:(i+1)*batch_size]

    latent_samples = make_latent_samples(batch_size, sample_size)
    X_batch_fake = generator.predict_on_batch(latent_samples)

    make_trainable(discriminator, True)
    discriminator.train_on_batch(X_batch_real, y_train_real)
    discriminator.train_on_batch(X_batch_fake, y_train_fake)

    make_trainable(discriminator, False)
    gan.train_on_batch(latent_samples, y_train_real)

  ##evaluation
  X_eval_real = X_test_real[np.random.choice(len(X_test_real), eval_size, replace=False)]
  latent_samples = make_latent_samples(eval_size, sample_size)
  X_eval_fake = generator.predict_on_batch(latent_samples)
