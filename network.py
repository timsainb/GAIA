import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
from tqdm import tqdm_notebook as tqdm

weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

class GAIA(object):
    def __init__(self, dims, batch_size, gpus = [], activation_fn = tf.nn.relu,
                 latent_loss = 'SSE', adam_eps = 1.0, network_type='AE',
                 n_res=4, n_sample=2, style_dim=8, ch=64, n_hidden = 512):

        self.dims = dims
        self.batch_size = batch_size
        self.latent_loss = latent_loss # either 'SSE', 'VAE', or 'distance'
        self.network_type = network_type
        # training loss for SSE and distance
        self.default_activation = activation_fn
        self.adam_eps = adam_eps

        self.n_res = n_res # number of residual layers
        self.n_sample = n_sample # number of resamples
        self.ch = ch # base number of filters
        self.img_ch = dims[2] # number of channels in image
        self.style_dim = style_dim
        self.mlp_dim = pow(2, self.n_sample) * self.ch  # default : 256
        self.n_downsample = self.n_sample
        self.n_upsample = self.n_sample
        self.n_hidden = n_hidden # how many hidden units in generator z

        self.num_gpus = len(gpus) # number of GPUs to use
        if len(gpus) < 1:
            self.num_gpus = 1

        self.initialize_network()

    def initialize_network(self):
        """ Defines the network architecture
        """
        # initialize graph and session
        self.graph = tf.Graph()
        self.config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.config.gpu_options.allocator_type = 'BFC'
        self.config.gpu_options.allow_growth=True
        self.sess = tf.InteractiveSession(graph=self.graph,config=self.config)

        # Global step needs to be defined to coordinate multi-GPU
        self.global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

        self.x_input = tf.placeholder(tf.float32, [self.batch_size*self.num_gpus, np.prod(self.dims)]) # Placeholder for input data

        if self.network_type == 'AE':
            self.AE_initialization()
        elif self.network_type == 'GAIA':
            self.GAIA_initialization()

        # apply the gradients with our optimizers

        # Start the Session
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver() # initialize network saver
        print('Network Initialized')


    def GAIA_initialization(self):
        """ Initialization specific to GAIA network
        """
        # run the x input through the network
        self.inference_GAIA(self.x_input)

        self.lr_sigma_slope = tf.placeholder('float32') # slope for balancing Generator and Descriminator in GAN
        self.lr_max = tf.placeholder('float32') # maximum learning rate for both networks

        ### RENAMING SOME PARAMETERS ###
        self.x_real = self.x_input
        #self.z_x = self.z_x_real
        self.x_tilde = self.x_real_recon
        self.latent_loss_weights = tf.placeholder(tf.float32) # Placeholder for weight of distance metric

        # distance loss
        self.distance_loss = distance_loss(self.x_input, self.z_gen_content_net_real)

        # compute losses of the model
        self.x_fake_from_real_recon_loss = tf.reduce_mean(tf.abs(self.x_real - self.x_fake_from_real_recon))
        self.x_fake_from_sample_recon_loss = tf.reduce_mean(tf.abs(self.x_fake_from_sample - self.x_fake_from_sample_recon))
        self.x_fake_recon_loss = (self.x_fake_from_sample_recon_loss + self.x_fake_from_real_recon_loss)/2.

        # compute losses of the model
        self.x_real_recon_loss = tf.reduce_mean(tf.abs(self.x_real - self.x_real_recon))

        # squash with a sigmoid based on the learning rate
        self.lr_D = sigmoid(self.x_real_recon_loss - self.x_fake_recon_loss, shift=0. , mult=self.lr_sigma_slope)
        self.lr_G = (tf.constant(1.0) - self.lr_D)*self.lr_max
        self.lr_D = self.lr_D*self.lr_max

        self.sigma = 0.5 # balance parameter for discriminator caring more about autoencoding real, or discriminating fake
        self.discrim_proportion_fake = tf.clip_by_value(
                    sigmoid(self.x_fake_from_sample_recon_loss*self.sigma- self.x_real_recon_loss, shift=0. , mult=self.lr_sigma_slope),
                     0., 0.9) # hold the discrim proportion fake aways at less than half
        self.discrim_proportion_real = tf.constant(1.)


        # add losses for generator and descriminator
        # loss of Encoder/Decoder: reconstructing x_real well and x_fake poorly
        self.L_d = tf.clip_by_value((self.x_real_recon_loss*self.discrim_proportion_real  - self.x_fake_recon_loss * self.discrim_proportion_fake), -1, 1)

        # hold the discrim proportion fake aways at less than half
        self.gen_proportion_sample = tf.clip_by_value(
                    sigmoid(self.x_fake_from_sample_recon_loss - self.x_fake_from_real_recon_loss, shift=0. , mult=self.lr_sigma_slope),
                     0., 1.0)

        # Generator should be balancing the reproduction
        self.L_g = tf.clip_by_value((self.gen_proportion_sample*self.x_fake_from_sample_recon_loss + \
                                    (1.0 - self.gen_proportion_sample)* self.x_fake_from_real_recon_loss) + \
                                    self.latent_loss_weights*self.distance_loss, \
                                    -1, 1)

        #M Global is just a way to visualize the training decrease
        self.m_global = self.x_real_recon_loss + self.x_fake_recon_loss

        # apply optimizers
        self.opt_D = tf.train.AdamOptimizer(learning_rate=self.lr_D, epsilon=self.adam_eps)
        self.opt_G = tf.train.AdamOptimizer(learning_rate=self.lr_G, epsilon=self.adam_eps)

        # specify loss to parameters
        self.params = tf.trainable_variables()

        self.D_params = [i for i in self.params if 'descriminator/' in i.name]
        self.G_params = [i for i in self.params if 'generator/' in i.name]

        # Calculate the gradients for the batch of data on this CIFAR tower.
        self.grads_d = self.opt_D.compute_gradients(self.L_d, var_list = self.D_params)
        self.grads_g = self.opt_G.compute_gradients(self.L_g, var_list = self.G_params)

        #
        self.train_D = self.opt_D.apply_gradients(self.grads_d, global_step=self.global_step)
        self.train_G = self.opt_G.apply_gradients(self.grads_g, global_step=self.global_step)

    def inference_GAIA(self, x_real):

        # Create a fake X value from input from the generator (this will try to autoencode it's input)
        print('Creating Generator...')
        with tf.variable_scope("generator"):
            with tf.variable_scope("enc"):
                print('...Creating encoder in generator...')
                self.gen_style_net_real, self.gen_content_net_real = self.encoder(self.x_input, discriminator = False)
                self.z_gen_style_net_real = self.gen_style_net_real[-1] # z value in the generator for the real image
                self.z_gen_content_net_real = self.gen_content_net_real[-1] # z value in the generator for the real image

            print('... Creating interpolations...')
            # get interpolation points as sampled from a gaussian distribution centered at 50%
            self.midpoint_div_1 = tf.random_normal(shape = (int(self.batch_size/2),), mean = 0.5, stddev = 0.25)
            self.midpoint_div_2 = tf.random_normal(shape = (int(self.batch_size/2),), mean = 0.5, stddev = 0.25)

            self.z_gen_style_net_sampled = get_midpoints(self.z_gen_style_net_real, self.midpoint_div_1, self.midpoint_div_2)
            self.z_gen_content_net_sampled = get_midpoints(self.z_gen_content_net_real, self.midpoint_div_1,self.midpoint_div_2)

            # run real images through the first autoencoder (the generator)
            print('...Creating decoder in generator...')
            with tf.variable_scope("dec"):
                self.gen_dec_net_from_real = self.decoder(self.z_gen_style_net_real, self.z_gen_content_net_real, discriminator = False) # fake generated image
                self.x_fake_from_real = self.gen_dec_net_from_real[-1]
            # run the sampled generator_z through the decoder of the generator
            with tf.variable_scope("dec", reuse=True):
                self.gen_dec_net_from_sampled = self.decoder(self.z_gen_style_net_sampled, self.z_gen_content_net_sampled, verbose=False, discriminator = False) # fake generated image
                self.x_fake_from_sample = self.gen_dec_net_from_sampled[-1]

        print('Creating Discriminator...')
        with tf.variable_scope("descriminator"):
            # Run the real x through the descriminator
            with tf.variable_scope("enc"):
                print('...Creating encoder in discriminator...')
                self.disc_style_net_real, self.disc_content_net_real = self.encoder(self.x_input, discriminator = True) # get z from the input
                self.z_disc_style_net_real = self.disc_style_net_real[-1] # z value in the descriminator for the real image
                self.z_disc_content_net_real = self.disc_content_net_real[-1] # z value in the descriminator for the real image

            with tf.variable_scope("dec"):
                print('...Creating decoder in discriminator...')
                self.disc_dec_net_real = self.decoder(self.z_disc_style_net_real, self.z_disc_content_net_real, discriminator = True) # get output from z
                self.x_real_recon = self.disc_dec_net_real[-1] # reconsruction of the real image in the descriminator

            # run the generated x which is autoencoding the real values through the network
            with tf.variable_scope("enc", reuse=True):
                self.disc_style_net_fake_from_real, self.disc_content_net_fake_from_real = self.encoder(self.x_fake_from_real, discriminator = True) # get z from the input
                self.z_disc_style_net_fake_from_real = self.disc_style_net_fake_from_real[-1] # z value in the descriminator for the real image
                self.z_disc_content_net_fake_from_real = self.disc_content_net_fake_from_real[-1] # z value in the descriminator for the real image

            with tf.variable_scope("dec", reuse=True):
                self.disc_dec_net_fake_from_real = self.decoder(self.z_disc_style_net_fake_from_real, self.z_disc_content_net_fake_from_real, verbose=False , discriminator = True) # get output from z
                self.x_fake_from_real_recon = self.disc_dec_net_fake_from_real[-1] # reconsruction of the fake image in the descriminator

            # run the interpolated (generated) x through the discriminator
            with tf.variable_scope("enc", reuse=True):
                self.disc_style_net_fake_from_sampled, self.disc_content_net_fake_from_sampled = self.encoder(self.x_fake_from_sample, discriminator = True) # get z from the input
                self.z_disc_style_net_fake_from_sampled = self.disc_style_net_fake_from_sampled[-1] # z value in the descriminator for the real image
                self.z_disc_content_net_fake_from_sampled = self.disc_content_net_fake_from_sampled[-1] # z value in the descriminator for the real image

            # run gen_x through the autoencoder, return the output
            with tf.variable_scope("dec", reuse=True):
                self.disc_dec_net_fake_from_sampled = self.decoder(self.z_disc_style_net_fake_from_sampled, self.z_disc_content_net_fake_from_sampled, verbose=False, discriminator = True) # get output from z
                self.x_fake_from_sample_recon = self.disc_dec_net_fake_from_sampled[-1] # reconsruction of the fake image in the descriminator



    def encoder(self, X, reuse=False, scope='content_encoder', verbose = False, discriminator = False):
        n_downsample = self.n_downsample
        channel = self.ch
        if discriminator == False:
            n_res = self.n_res
        else:
            n_res = self.n_res# - 2
            channel = channel/2

        ###### Content Encoder
        with tf.variable_scope("content_enc"):
            content_net = [tf.reshape(X, [self.batch_size, self.dims[0], self.dims[1], self.dims[2]])]
            content_net.append(relu(conv(content_net[len(content_net)-1], channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')))

            for i in range(n_downsample):
                content_net.append(relu(conv(content_net[len(content_net)-1], channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i + 1))))
                channel = channel * 2

            for i in range(n_res):
                content_net.append(resblock(content_net[len(content_net)-1], channel, scope='resblock_' + str(i)))


            if discriminator == False:
                content_net.append(linear(content_net[len(content_net)-1], self.n_hidden, use_bias=True, scope='linear'))
            content_shapes =  [shape(i) for i in content_net]

            if verbose: print('content_net shapes: ',content_shapes)
        if discriminator == False:
            channel = self.ch
        else:
            channel = self.ch/2

        ###### Style Encoder
        with tf.variable_scope("style_enc"):

            # IN removes the original feature mean and variance that represent important style information
            style_net = [tf.reshape(X, [self.batch_size, self.dims[0], self.dims[1], self.dims[2]])]
            style_net.append(conv(style_net[len(style_net)-1], channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0'))
            style_net.append(relu(style_net[len(style_net)-1]))

            for i in range(2):
                style_net.append(relu(conv(style_net[len(style_net)-1], channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i + 1))))
                channel = channel * 2

            for i in range(2):
                style_net.append(relu(conv(style_net[len(style_net)-1], channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='down_conv_' + str(i))))

            style_net.append(adaptive_avg_pooling(style_net[len(style_net)-1]))  # global average pooling
            style_net.append(conv(style_net[len(style_net)-1], self.style_dim, kernel=1, stride=1, scope='SE_logit'))
            style_shapes =  [shape(i) for i in style_net]
            if verbose: print('style_net shapes: ',style_shapes)

        return style_net, content_net

    def decoder(self, z_x_style, z_x_content, reuse=False, scope="content_decoder", verbose=False, discriminator = False):
        channel = self.mlp_dim
        n_upsample = self.n_upsample
        if discriminator == False:
            n_res = self.n_res
            z_x_content = tf.reshape(linear(z_x_content, (self.dims[0]/(2**self.n_sample))*(self.dims[0] /(2**self.n_sample))*(self.ch * self.n_sample**2)), (self.batch_size, int(self.dims[0] /(2**self.n_sample)), int(self.dims[0] /(2**self.n_sample)), int(self.ch * self.n_sample**2)))
        else:
            n_res = self.n_res# - 2
            channel = channel /2
        dec_net = [z_x_content]
        mu, sigma = self.MLP(z_x_style, discriminator= discriminator)
        for i in range(n_res):
            dec_net.append(adaptive_resblock(dec_net[len(dec_net)-1], channel,mu, sigma, scope='adaptive_resblock' + str(i)))

        for i in range(n_upsample):
            # # IN removes the original feature mean and variance that represent important style information
            dec_net.append(up_sample(dec_net[len(dec_net)-1], scale_factor=2))
            dec_net.append(conv(dec_net[len(dec_net)-1], channel // 2, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_' + str(i)))
            dec_net.append(relu(layer_norm(dec_net[len(dec_net)-1], scope='layer_norm_' + str(i))))

            channel = channel // 2
        dec_net.append(conv(dec_net[len(dec_net)-1], channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit'))
        dec_net.append(tf.reshape(tf.sigmoid(dec_net[len(dec_net)-1]), [self.batch_size, self.dims[0]*self.dims[1]*self.dims[2]]))
        dec_shapes =  [shape(i) for i in dec_net]
        if verbose: print('Decoder shapes: ', dec_shapes)
        return dec_net

    def MLP(self, style, reuse=False, scope='MLP', discriminator = False):
        channel = self.mlp_dim
        if discriminator:
            channel = int(channel/2)
        with tf.variable_scope(scope, reuse=reuse):
            x = relu(linear(style, channel, scope='linear_0'))
            x = relu(linear(x, channel, scope='linear_1'))

            mu = linear(x, channel, scope='mu')
            sigma = linear(x, channel, scope='sigma')

            mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            sigma = tf.reshape(sigma, shape=[-1, 1, 1, channel])
            return mu, sigma


    def _get_tensor_by_name(self, tensor_list):
        return [self.graph.get_tensor_by_name(i) for i in tensor_list]

    def save_network(self, save_location, verbose=True):
        """ Save the network to some location"""
        self.saver.save(self.sess,''.join([save_location]))
        if verbose: print('Network Saved')

    def load_network(self, load_location, verbose=True):
        """ Retrieve the network from some location"""
        self.saver = tf.train.import_meta_graph(load_location + '.meta')
        self.saver.restore(self.sess,tf.train.latest_checkpoint('/'.join(load_location.split('/')[:-1]) +'/'))
        if verbose: print('Network Loaded')
    def encode(self, X, only_z=False):
        """encode input into z values and output"""
        return self.sess.run((self.x_tilde, self.z_gen_style_net_real), {self.x_input: X})



def norm(X):
    return (X - np.min(X))/(np.max(X)-np.min(X))

def norm_ax(X):
    return X / X.max(axis=1)[:, np.newaxis]

def z_score(X):
    return (X - np.mean(X))/np.std(X)

def inv_z_score(Y, X):
    return (Y * np.std(X)) + np.mean(X)

def shape(tensor):
    """ get the shape of a tensor
    """
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def squared_dist(A):
    """
    Computes the pairwise distance between points
    #http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    """
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_mean(tf.squared_difference(expanded_a, expanded_b), 2)
    return distances


def MSE_loss(x, y):
    """ Classic mean squared error loss
    """
    return tf.reduce_mean(tf.square(x - y))

def L1_loss(x, y):
    """ Classic mean squared error loss
    """
    return tf.reduce_mean(tf.abs(x - y))

def distance_loss(x, z_x):
    """ Loss based on the distance between elements in a batch
    """
    z_x = tf.reshape(z_x, [shape(z_x)[0], np.prod(shape(z_x)[1:])])
    sdx = squared_dist(x)
    sdx = sdx/tf.reduce_mean(sdx)
    sdz = squared_dist(z_x)
    sdz = sdz/tf.reduce_mean(sdz)
    return tf.reduce_mean(tf.square(tf.log(tf.constant(1.)+sdx) - (tf.log(tf.constant(1.)+sdz))))

def distance_loss_true(x, z_x):
    """ Loss based on the distance between elements in a batch
    """
    sdx = squared_dist(x)
    sdz = squared_dist(z_x)
    return tf.reduce_mean(tf.abs(sdz - sdx))

def KL_loss(z_x_mean, z_log_sigma_sq):
    """ Loss function used in Variational Autoencoders
    """
    return tf.reduce_sum(-0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_x_mean)
                                           - tf.exp(z_log_sigma_sq), 1))
def sigmoid(x,shift,mult):
    return tf.constant(1.) / (tf.constant(1.)+ tf.exp(-tf.constant(1.0)*(x*mult)))



def create_image(im, dims):
    return np.reshape(im,(dims[0],dims[1]))



def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if scope.__contains__("discriminator") :
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else :
            weight_init = tf_contrib.layers.variance_scaling_initializer()

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)

        return x + x_init

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    return gap

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def get_midpoints(z, midpoint_div_1, midpoint_div_2):
    """ Gets this midpoints in Z space between a batch of images
    """
    # reshape z
    shape_z = shape(z)
    z = tf.reshape(z, [shape_z[0], np.prod(shape_z[1:])])
    hidden_size = np.prod(shape_z[1:])
    batch_size = shape(z)[0]
    # get the first half of the batch
    first_half_batch = tf.slice(z, [0,0], [int(batch_size/2), hidden_size])
    # get the second half of the batch
    second_half_batch = tf.slice(z, [int(shape(z)[0]/2),0], [int(shape(z)[0]/2), hidden_size])
    # get the midpoint as image_1*midpoint_div_1 + image2 * (1- midpoint_div_1)
    midpoints_1 = tf.multiply(first_half_batch, tf.expand_dims(midpoint_div_1,1)) + tf.multiply(second_half_batch, tf.expand_dims(1. - midpoint_div_1,1))
    # for the second interpolated group, flip the ordering of the first half of the batch so we use new interpolations
    midpoints_2 = tf.multiply(tf.reverse(first_half_batch, [0]), tf.expand_dims(midpoint_div_2,1)) + tf.multiply(second_half_batch, tf.expand_dims(1. - midpoint_div_2,1))
    z_sampled = (tf.concat(axis = 0, values = [midpoints_1, midpoints_2]))
    z_sampled = tf.reshape(z_sampled, shape_z)
    return z_sampled
