import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist


OUT_DIR = 'DNN_out'
img_shape = (28,28,1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)

X_train = X_train / 127.5 - 1  # 데이터 범위를 -1 ~ 1로 한정
X_train = np.expand_dims(X_train, axis=3) # 차원을 하나 늘림. reshape나 마찬가지.
print(X_train.shape)

generator = Sequential()
generator.add(Dense(256, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01))
generator.add(BatchNormalization(momentum=0.8))  # 배치 정규화 추가
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=0.01))
generator.add(BatchNormalization(momentum=0.8))  # 배치 정규화 추가
generator.add(Dense(1024))
generator.add(LeakyReLU(alpha=0.01))
generator.add(BatchNormalization(momentum=0.8))  # 배치 정규화 추가
generator.add(Dense(np.prod(img_shape), activation='tanh'))
generator.add(Reshape(img_shape))
generator.summary()

lrelu = LeakyReLU(alpha=0.01)
discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    z = np.random.normal(0,1, (batch_size, noise))
    fake_img = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator.train_on_batch(fake_img, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)
    discriminator.trainable=False
    if epoch % 4 == 0:
        z = np.random.normal(0, 1, (batch_size, noise))
        gan_hist = gan_model.train_on_batch(z, real)

    if epoch % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(epoch, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        count = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1))
        plt.savefig(path)
        plt.close()

        gan_model.trainable = True
        gan_model.save('./models/GAN.h5')
        discriminator.trainable = True
        discriminator.save('./models/discriminator.h5')
