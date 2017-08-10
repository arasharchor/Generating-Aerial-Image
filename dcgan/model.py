from dcgan.utils import *
import matplotlib.image as mp


class DCGAN:
    def __init__(self, in_dim, z_dim, depth, learning_rate, beta1, batch_size=64, k=1,
                 add_noise_to_D=True, training_epochs=20000, using_gpu=False):
        """
        in_dim : list ,[height , width , channel]
        z_dim  : int  ,dimension of generator noise
        depth  : list ,depth of each convolution
        learning_rate : float , learning rate for discriminator's and generator's optimizer
        beta1  : float , adam
        k : number of training discriminator in each iteration
        """

        self.in_dim = in_dim
        self.z_dim = z_dim
        self.depth = depth
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.batch_size = batch_size
        self.k = k
        self.add_noise_to_D = add_noise_to_D
        self.training_epochs = training_epochs
        self.using_gpu = using_gpu
        self.tensorboard_dir = '../log/gan/'
        self.summaries_path = '../log/summaries/'
        self.model_save_path = '../log/model/model.ckpt'
        self.sess = tf.Session()
        self.D_loss_threshold = 0.2
        self.G_loss_threshold = 0.3
        self.g_additional_training_count = 2
        self.d_additional_training_count = 2
        self.D_MAX_ERROR = 0.7
        self.G_MAX_ERROR = 0.7

    def generator(self, noise):
        with tf.variable_scope('generator') as scope:
            # 100 * 1
            output = multiply(noise, [self.z_dim, 7 * 7 * 256], name='fc')

            output = tf.reshape(output, (-1, 7, 7, 256))
            # 7 * 7 * 256
            output = deconv2d(output, [3, 3, 1024, 256], [self.batch_size, 14, 14, 1024], 2, name='deconv1')
            # 14 * 14 * 1024
            output = deconv2d(output, [3, 3, 512, 1024], [self.batch_size, 28, 28, 512], 2, name='deconv2')
            # 28 * 28 * 512
            output = deconv2d(output, [3, 3, 256, 512], [self.batch_size, 56, 56, 256], 2, name='deconv3')
            # 56 * 56 * 256
            output = deconv2d(output, [3, 3, 3, 256], [self.batch_size, 112, 112, 3], 2, name='deconv4', use_tanh=True)
            # 112 * 112 * 3
            return output

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            # 112 * 112 * 3
            output = conv2d(x, [3, 3, 3, 256], name='conv1')
            # 56 * 56 * 256
            output = conv2d(output, [3, 3, 256, 512], name='conv2')
            # 28 * 28 * 512
            output = conv2d(output, [3, 3, 512, 1024], name='conv3')
            # 14 * 14 * 1024
            output = conv2d(output, [3, 3, 1024, 256], name='conv4')
            # 7 * 7 * 256
            output = tf.reshape(output, (-1, 7 * 7 * 256))

            output = multiply(output, [7 * 7 * 256, 8370], name='fc1')

            output = multiply(output, [8370, 4096], name='fc2')

            output = multiply(output, [4096, 1], name='output_layer', activation='linear')

            return output

    def train(self, images, restore=False):
        I = tf.placeholder(dtype=tf.float32,
                           name='images',
                           shape=[None, self.in_dim, self.in_dim, 3])

        noise = sample_noise([self.batch_size, self.z_dim])
        I_ = self.generator(noise)
        real_detection = self.discriminator(I)
        fake_detection = self.discriminator(I_, reuse=True)

        with tf.name_scope('Loss'):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_detection), logits=real_detection)
            )
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_detection), logits=fake_detection)
            )
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_detection), logits=fake_detection)
            )

            d_loss = d_loss_fake + d_loss_real


        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

        with tf.name_scope('Optimization'):
            optimize_d = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
            optimize_g = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

        init = tf.global_variables_initializer()

        with tf.name_scope('summaries'):
            tf.summary.image('generated_images', I_)
            tf.summary.scalar('d_loss_real', d_loss_real)
            tf.summary.scalar('d_loss_fake', d_loss_fake)
            tf.summary.scalar('g_loss', g_loss)
            merged = tf.summary.merge_all()
        step = 0

        with self.sess:
            print(colored('Training Started...', 'green'))
            summary_writer = tf.summary.FileWriter(self.summaries_path, self.sess.graph)
            saver = tf.train.Saver()
            if not restore:
                self.sess.run(init)
            else:
                saver.restore(self.sess, self.model_save_path)

            for epoch in range(self.training_epochs):

                for image_batch in self.get_batch(images):

                    for _ in range(self.k):
                        # train Discriminator
                        self.sess.run([optimize_d],
                                      feed_dict={I: image_batch})

                    # train Generator
                    self.sess.run([optimize_g],
                                  feed_dict={I: image_batch})

                    d_cost, g_cost = self.sess.run([d_loss, g_loss],
                                                   feed_dict={I: image_batch})

                    if d_cost < self.D_loss_threshold and abs(d_cost - g_cost) > 0.2:
                        counter = 0
                        while abs(d_cost - g_cost) > 0.2:
                            random_batch = self.get_random_batch(images)
                            _, g_cost, d_cost = self.sess.run([optimize_g, g_loss, d_loss],
                                                              feed_dict={I: random_batch})
                            counter += 1
                            if counter >= 2:
                                break

                    if g_cost < self.G_loss_threshold and abs(g_cost - d_cost) > 0.2:
                        counter = 0
                        while abs(d_cost - g_cost) > 0.2:
                            random_batch = self.get_random_batch(images)
                            _, g_cost, d_cost = self.sess.run([optimize_d, g_loss, d_loss],
                                                              feed_dict={I: random_batch})
                            counter += 1
                            if counter >= 2:
                                break

                    if g_cost > self.G_MAX_ERROR:

                        while g_cost > self.G_MAX_ERROR:
                            random_batch = self.get_random_batch(images)
                            _, g_cost = self.sess.run([optimize_g, g_loss], feed_dict={I: random_batch})

                    if d_cost > self.D_MAX_ERROR:

                        while d_cost > self.D_MAX_ERROR:
                            random_batch = self.get_random_batch(images)
                            _, d_cost = self.sess.run([optimize_d, d_loss], feed_dict={I: random_batch})

                    if step % 5 == 0:
                        generated_images = self.sess.run(I_)
                        mp.imsave('../generated_images/' + str(step) + '.jpeg', generated_images[0])
                        summary = self.sess.run(merged, feed_dict={I: image_batch})
                        summary_writer.add_summary(summary, step)

                    if step % 100 == 0 and step > 0:
                        saver.save(self.sess, self.model_save_path)
                    step += 1
                    print('G cost:', colored(g_cost, 'green'), '  ', 'D cost:',
                          colored(d_cost, 'red'))
                    print(colored('-------------------------', 'blue'))

    def get_batch(self, images):

        num_of_batch = int(len(images) / self.batch_size)
        for i in range(num_of_batch):
            yield images[i * self.batch_size:(i + 1) * self.batch_size]

    def get_random_batch(self, images):
        num_of_batch = int(len(images) / self.batch_size)
        random_index = np.random.randint(0, num_of_batch - 2)
        return images[random_index * self.batch_size: (random_index + 1) * self.batch_size]
