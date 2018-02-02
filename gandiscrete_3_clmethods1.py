from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv

import tensorflow as tf

# import generate_data as gd


def simplyplot(filename):
    plt.style.use('ggplot')
    original = pd.read_csv(filename)
    # original = gd.genran2()




    #----------------------------------------------
    scale = 2.5
    fig, axs = plt.subplots(2, 3, figsize=(9*scale, 3*scale), sharey=True)
    # original.plot.bar(stacked=True)
    # original.plot.hist(stacked=True)
    # ser = pd.Series(original['A'])
    # original['A'].value_counts(sort=False).plot(ax = axs[0,0],kind ='bar')
    # original['B'].value_counts(sort=False).plot(ax = axs[0,1],kind ='bar')
    # original['C'].value_counts(sort=False).plot(ax = axs[0,2],kind ='bar')
    # original['D'].value_counts(sort=False).plot(ax = axs[1,0],kind ='bar')
    # original['E'].value_counts(sort=False).plot(ax = axs[1,1],kind ='bar')
    # original['F'].value_counts(sort=False).plot(ax = axs[1,2],kind ='bar')


    #----------------------------------------------

    original['A'].plot(ax = axs[0,0],kind ='hist')
    original['B'].plot(ax = axs[0,1],kind ='hist')
    original['C'].plot(ax = axs[0,2],kind ='hist')
    original['D'].plot(ax = axs[1,0],kind ='hist')
    original['E'].plot(ax = axs[1,1],kind ='hist')
    original['F'].plot(ax = axs[1,2],kind ='hist')

    fig.suptitle(filename)

    plt.show()
    #----------------------------------------------


    # original.plot(kind = 'density')
    #
    #
    # plt.show()

class mydataset(object):

    def __init__(self, dataset ):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(dataset)
        self.images = dataset
        self._images = dataset

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples).astype('uint32')
            np.random.shuffle(perm0)
            self._images = np.array(self.images)[perm0].tolist()

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            # labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = np.array(self.images)[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = np.array(self.images)[start:end]

            return np.concatenate((images_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end]



def gan_org(original, splitpoint = 0.9, outputfilename='data/test4.csv'):




    slpt = (int)(splitpoint * original.shape[0])

    # training_set = original.iloc[0:slpt,0:5].values.tolist()
    # training_target = original.iloc[0:slpt,-1].values.tolist()
    # test_set = original.iloc[slpt:original.shape[0],0:5].values.tolist()
    # test_target = original.iloc[slpt:original.shape[0], -1].values.tolist()

    training_set = original.iloc[0:slpt, :].values.tolist()
    test_set = original.iloc[slpt:original.shape[0], :].values.tolist()


    mydata = mydataset(training_set)


    # Training Params
    num_steps = 3000
    batch_size = 128
    learning_rate = 0.0002

    # Network Params
    image_dim = original.shape[1]  # 28*28 pixels
    gen_hidden_dim1 = 100
    gen_hidden_dim2 = 200
    gen_hidden_dim3 = 100
    disc_hidden_dim1 = 100
    disc_hidden_dim2 = 200
    disc_hidden_dim3 = 100
    noise_dim = int(original.shape[1] * 0.9)  # Noise data points
    # noise_dim = original.shape[1]  # Noise data points

    # A custom initialization (see Xavier Glorot init)
    def glorot_init(shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    # Store layers weight & bias
    weights = {
        'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim1])),
        'gen_hidden2': tf.Variable(glorot_init([gen_hidden_dim1, gen_hidden_dim2])),
        'gen_hidden3': tf.Variable(glorot_init([gen_hidden_dim2, gen_hidden_dim3])),
        'gen_out': tf.Variable(glorot_init([gen_hidden_dim3, image_dim])),

        'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim1])),
        'disc_hidden2': tf.Variable(glorot_init([disc_hidden_dim1, disc_hidden_dim2])),
        'disc_hidden3': tf.Variable(glorot_init([disc_hidden_dim2, disc_hidden_dim3])),
        'disc_out': tf.Variable(glorot_init([disc_hidden_dim3, 1])),
    }
    biases = {
        'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim1])),
        'gen_hidden2': tf.Variable(tf.zeros([gen_hidden_dim2])),
        'gen_hidden3': tf.Variable(tf.zeros([gen_hidden_dim3])),
        'gen_out': tf.Variable(tf.zeros([image_dim])),

        'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim1])),
        'disc_hidden2': tf.Variable(tf.zeros([disc_hidden_dim2])),
        'disc_hidden3': tf.Variable(tf.zeros([disc_hidden_dim3])),
        'disc_out': tf.Variable(tf.zeros([1])),
    }

    # Generator
    def generator(x):
        hidden_layer1 = tf.matmul(x, weights['gen_hidden1'])
        hidden_layer1 = tf.add(hidden_layer1, biases['gen_hidden1'])
        hidden_layer1 = tf.nn.relu(hidden_layer1)

        hidden_layer2 = tf.matmul(hidden_layer1, weights['gen_hidden2'])
        hidden_layer2 = tf.add(hidden_layer2, biases['gen_hidden2'])
        hidden_layer2 = tf.nn.relu(hidden_layer2)

        hidden_layer3 = tf.matmul(hidden_layer2, weights['gen_hidden3'])
        hidden_layer3 = tf.add(hidden_layer3, biases['gen_hidden3'])
        hidden_layer3 = tf.nn.relu(hidden_layer3)

        out_layer = tf.matmul(hidden_layer3, weights['gen_out'])
        out_layer = tf.add(out_layer, biases['gen_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    # Discriminator
    def discriminator(x):
        hidden_layer1 = tf.matmul(x, weights['disc_hidden1'])
        hidden_layer1 = tf.add(hidden_layer1, biases['disc_hidden1'])
        hidden_layer1 = tf.nn.relu(hidden_layer1)

        hidden_layer2 = tf.matmul(hidden_layer1, weights['disc_hidden2'])
        hidden_layer2 = tf.add(hidden_layer2, biases['disc_hidden2'])
        hidden_layer2 = tf.nn.relu(hidden_layer2)

        hidden_layer3 = tf.matmul(hidden_layer2, weights['disc_hidden3'])
        hidden_layer3 = tf.add(hidden_layer3, biases['disc_hidden3'])
        hidden_layer3 = tf.nn.relu(hidden_layer3)


        out_layer = tf.matmul(hidden_layer3, weights['disc_out'])
        out_layer = tf.add(out_layer, biases['disc_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    # Build Networks
    # Network Inputs
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

    # Build Generator Network
    gen_sample = generator(gen_input)

    # Build 2 Discriminator Networks (one from noise input, one from generated samples)
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_sample)

    # Build Loss
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    # Build Optimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    gen_vars = [weights['gen_hidden1'], weights['gen_hidden2'],weights['gen_hidden3'],weights['gen_out'],
                biases['gen_hidden1'], biases['gen_hidden2'],biases['gen_hidden3'],biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [weights['disc_hidden1'], weights['disc_hidden2'], weights['disc_hidden3'], weights['disc_out'],
                 biases['disc_hidden1'], biases['disc_hidden2'], biases['disc_hidden3'], biases['disc_out']]

    # Create training operations
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for i in range(1, num_steps + 1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x = mydata.next_batch(batch_size)
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Train
            feed_dict = {disc_input: batch_x, gen_input: z}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        with open(outputfilename+'_real.csv', 'wb') as f:
            # f.write(b'A,B,C,D,E,F\n')
            np.savetxt(outputfilename+'_real.csv', np.array(test_set), delimiter=',', fmt='%.20f')

        numtestcases = len(test_set)
        # numtestcases = int(len(test_set) / 2)
        print('gan is generating ' + str(numtestcases) + ' records...')
        # for i in range(numtestcases):
            # Noise input.
        z = np.random.uniform(-1., 1., size=[numtestcases, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        with open(outputfilename, 'wb') as f:
            # f.write(b'A,B,C,D,E,F\n')
            np.savetxt(outputfilename, g[0], delimiter=',', fmt='%.20f')
            return g[0].shape
            # g = np.reshape(g, newshape=(4, 28, 28, 1))
            # # Reverse colours for better display
            # g = -1 * (g - 1)
            # for j in range(4):
            #     # Generate image from noise. Extend to 3 channels for matplot figure.
            #     img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
            #                      newshape=(28, 28, 3))
            #     a[j][i].imshow(img)



        # # Generate images from noise, using the generator network.
        # f, a = plt.subplots(4, 10, figsize=(10, 4))
        # for i in range(10):
        #     # Noise input.
        #     z = np.random.uniform(-1., 1., size=[4, noise_dim])
        #     g = sess.run([gen_sample], feed_dict={gen_input: z})
        #     g = np.reshape(g, newshape=(4, 28, 28, 1))
        #     # Reverse colours for better display
        #     g = -1 * (g - 1)
        #     for j in range(4):
        #         # Generate image from noise. Extend to 3 channels for matplot figure.
        #         img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
        #                          newshape=(28, 28, 3))
        #         a[j][i].imshow(img)
        #
        # f.show()
        # plt.draw()
        # plt.waitforbuttonpress()
        # simplyplot('data/test3_real.csv')
        # simplyplot('data/test3.csv')


def main():
    # original = pd.read_csv('data/test2.csv', float_precision='%.3f')
    original = pd.read_csv('data/adult.data', float_precision='%.3f')
    gan_org(original, splitpoint = 0.9)


if __name__ == "__main__":
    # def kl_divergence(p, q):
    #     sess = tf.Session()
    #     return sess.run(tf.reduce_sum(p * tf.log(p / q)))
    # print(tf.distributions.kl_divergence(val, test_target))
    # print(kl_divergence(tf.constant([1.0,2,3]), tf.constant([1.0,2,3])))
    # simplyplot('data/test3_real.csv')
    main()



