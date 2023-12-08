from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_docs.vis import embed
import tensorflow as tf
import numpy as np
import imageio

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128
epochs = 30

# 데이터 로드
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x데이터와 y데이터를 각각 하나의 배열로 만듬
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# 정규화
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# tf.data.Dataset으로 만듬
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

# 판별자
discriminator = keras.Sequential(name = "discriminator")
discriminator.add(keras.layers.InputLayer((28, 28, discriminator_in_channels)))
discriminator.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
discriminator.add(layers.LeakyReLU(alpha=0.2))
discriminator.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
discriminator.add(layers.LeakyReLU(alpha=0.2))
discriminator.add(layers.GlobalMaxPooling2D())
discriminator.add(layers.Dense(1))


# 생성자
generator = keras.Sequential(name = "generator")
generator.add(keras.layers.InputLayer((generator_in_channels,)))
generator.add(layers.Dense(7 * 7 * generator_in_channels))
generator.add(layers.LeakyReLU(alpha=0.2))
generator.add(layers.Reshape((7, 7, generator_in_channels)))
generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
generator.add(layers.LeakyReLU(alpha=0.2))
generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
generator.add(layers.LeakyReLU(alpha=0.2))
generator.add(layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"))

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # 데이터를 언팩
        real_images, one_hot_labels = data

        # 판별자를 위해 레이블에 더미차원을 추가하여 이미지와 합칠 수 있게 함.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # 생성자를 위해 latent space에서 무작위 포인트를 샘플링하고 레이블과 합침
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # 노이즈를 디코드하여 (레이블에 따라) 가짜 이미지를 생성합니다.
        generated_images = self.generator(random_vector_labels)

        # 이를 실제 이미지와 결합. 여기서 레이블이 이미지와 합쳐짐.
        # Combine them with real images. Note that we are concatenating the labels
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # 가짜 이미지들을 판별하는 레이블을 어셈블
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # 판별자 훈련
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # 잠재 공간에서 무작위 포인트를 샘플링
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # 전부 실제 이미지라고 말하는 레이블을 어셈블
        misleading_labels = tf.zeros((batch_size, 1))

        # 생성자 훈련 (판별자의 가중치를 업뎃해선 안됨)
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # 손실률을 모니터링
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

cond_gan = ConditionalGAN(
    discriminator=discriminator,
    generator=generator,
    latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, epochs=20)
cond_gan.save('./my_cGANs.h5')

# 조건부 GAN으로 훈련된 생성자를 추출
trained_gen = cond_gan.generator

# 보간 사이에 생성될 중간 이미지의 수를 선택
# +2는 시작이미지와 마지막 이미지를 위해 존재함
num_interpolation = 9  # @param {type:"integer"}


# 보간을 위한 노이즈를 샘플링
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))

def interpolate_class(first_number, second_number):
    # 시작 레이블과 끝 레이블을 원-핫 인코딩된 벡터로 변환
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # 두 레이블 사이의 보간 벡터를 계산
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # 노이즈와 레이블을 결합하고 생성자를 사용하여 추론을 실행
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake

start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
end_class = 5  # @param {type:"slider", min:0, max:9, step:1}

fake_images = interpolate_class(start_class, end_class)

# 이미지를 255로 스케일링
fake_images *= 255.0
# 이미지를 uint8 형식으로 변환
converted_images = fake_images.astype(np.uint8)
# 이미지 크기를 조정
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
# 이미지를 GIF로 저장
imageio.mimsave("./animation.gif", converted_images, fps=1)
# GIF를 임베드
embed.embed_file("./animation.gif")