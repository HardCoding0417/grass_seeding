from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import tensorflow as tf
import numpy as np
import imageio

"""
## 상수 및 하이퍼 파라미터
"""

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

"""
## MNIST 데이터 로딩 및 전처리
"""

# 훈련 세트와 테스트 세트에서 사용 가능한 모든 예제를 사용
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# 픽셀 값을 [0, 1] 범위로 조정하고, 이미지에 채널 차원을 추가하며, 레이블을 원-핫 인코딩
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# tf.data.Dataset을 생성
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"훈련 이미지의 형태: {all_digits.shape}")
print(f"훈련 레이블의 형태: {all_labels.shape}")

"""
## 생성기와 판별기의 입력 채널 수 계산

일반적인 (비조건부) GAN에서는 일정한 차원의 노이즈를 정규 분포에서 샘플링하여 시작함 
이 경우에는 클래스 레이블도 고려해야 하므로
생성기(노이즈 입력)와 판별기(생성된 이미지 입력)의 입력 채널에 클래스 수를 추가
"""

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

"""
## 판별기와 생성기 생성

모델 정의(`discriminator`, `generator`, `ConditionalGAN`)는
[이 예제](https://keras.io/guides/customizing_what_happens_in_fit/)에서 가져왔다
"""

# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

"""
## `ConditionalGAN` 모델 생성
"""


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
        # 데이터를 분해.
        real_images, one_hot_labels = data

        # 판별기 - 레이블에 더미 차원을 추가하여 이미지와 연결
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # 생성기 - 잠재 공간에서 무작위 점을 샘플링하고 레이블과 연결
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # 노이즈(레이블에 의해 안내됨)를 가짜 이미지로 디코딩
        generated_images = self.generator(random_vector_labels)

        # 가짜 이미지와 실제 이미지를 결합. 여기서는 레이블을 이 이미지들과 연결
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # 실제 이미지와 가짜 이미지를 구별하는 레이블을 조립.
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

        # latent space에 랜덤 포인트를 샘플함
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # 모두 진짜 이미지라고 말하는 레이블을 어셈블
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # 손실을 모니터링합니다.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

"""
## 조건부 GAN 훈련
"""

cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, epochs=1)


# # 모델 저장
# try:
#     cond_gan.save('model.h5')
# except:
#     cond_gan.save('model', save_format='tf')
#     cond_gan.save_weights('model_weights.h5')

"""
## 시연을 위해 훈련된 생성기로 클래스 간 보간하기
"""

# 먼저 조건부 GAN에서 훈련된 생성기를 추출합니다.
trained_gen = cond_gan.generator

# 보간 사이에 생성될 중간 이미지의 수를 선택합니다 + 2 (시작 및 마지막 이미지).
num_interpolation = 9  # @param {type:"integer"}

# 보간을 위한 노이즈를 샘플링합니다.
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))

def interpolate_class(first_number, second_number):
    # 시작 및 끝 레이블을 원-핫 인코딩된 벡터로 변환합니다.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # 두 레이블 사이의 보간 벡터를 계산합니다.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
            first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # 노이즈와 레이블을 결합하고 생성기로 추론을 실행합니다.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake

start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
end_class = 5  # @param {type:"slider", min:0, max:9, step:1}

fake_images = interpolate_class(start_class, end_class)

"""
여기서 우리는 먼저 정규 분포에서 노이즈를 샘플링한 다음, 
이를 `num_interpolation`만큼 반복하여 결과를 적절히 재구성합니다.
그런 다음 `num_interpolation`에 대해 균일하게 분포시키고, 레이블 신원이 일정 비율로 존재하게 합니다.
"""

fake_images *= 255.0
converted_images = fake_images.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("./animation.gif", converted_images, fps=1)
embed.embed_file("./animation.gif")

"""
이 모델의 성능은 [WGAN-GP](https://keras.io/examples/generative/wgan_gp)와 
같은 레시피로 더욱 향상시킬 수 있습니다.
조건부 생성은 [VQ-GANs](https://arxiv.org/abs/2012.09841), 
[DALL-E](https://openai.com/blog/dall-e/) 등 많은 현대 이미지 생성 아키텍처에서도 널리 사용됩니다.

[Hugging Face Hub](https://huggingface.co/keras-io/conditional-gan)에 호스팅된 훈련된 모델을 사용하거나 
[Hugging Face Spaces](https://huggingface.co/spaces/keras-io/conditional-GAN)에서 
데모를 시도해 볼 수 있습니다.
"""

