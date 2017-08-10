from dcgan.model import DCGAN
from dcgan.utils import *

images_dir = '../images/'

images = read_image(images_dir, 'jpeg')
model = DCGAN(in_dim=112,
              z_dim=100,
              depth=[256, 1024, 512, 256, 3],
              learning_rate=0.0001,
              beta1=0.8,
              batch_size=64,
              k=2,
              add_noise_to_D=False,
              training_epochs=200000,
              using_gpu=True)
model.train(images)
