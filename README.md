# GAN-style
Generating art combining various artists using different styles of GAN
Source dataset: !kaggle datasets download -d ikarus777/best-artworks-of-all-time (2GB)

![image](https://user-images.githubusercontent.com/40152283/235613959-23b76d10-23b3-464a-b5d7-ade9ffc2d772.png)

Begins with a simple GAN using pytorch and tensorflow.

**The Generator:**

The generator and discriminator in a Generative Adversarial Network (GAN) are designed to work together in a two-player minimax game, 
where the generator creates fake images and the discriminator distinguishes between real and fake images. 
The generator aims to produce images that the discriminator cannot distinguish from real ones, while the discriminator aims to correctly 
identify real and fake images. Over time, the generator learns to create increasingly realistic images, and the discriminator improves its 
ability to identify fakes.

The generator is designed to map a random noise vector (latent space) to an image with the same dimensions as the training data. The architecture consists of a series of transposed convolutional layers (also known as deconvolutional layers) which progressively upscale the input noise vector to the desired output image size. The use of transposed convolution allows the generator to learn spatial hierarchies and capture fine-grained details in the generated images.

Batch normalization is used after most layers to stabilize training and improve convergence by reducing the internal covariate shift. LeakyReLU activation functions are used instead of regular ReLU activation to mitigate the vanishing gradient problem, allowing the generator to learn more effectively. The final layer uses a tanh activation function to ensure that the output values are in the range of [-1, 1], which is the typical range for image pixel values after normalization.

z_dim: The dimension of the input noise vector (default is 10).
in_chan: The number of input channels (default is 3, for RGB images).
hidden_dim: The dimension of the hidden layers (default is 64).

The generator architecture is a deep convolutional neural network (CNN) composed of a series of layers:

A generator block that maps the input noise vector to a 1x1x512 tensor.

Five blocks of transposed convolutional layers, each followed by batch normalization and LeakyReLU activation.
A final generator block with the same number of output channels as in_chan and a tanh activation function.

The make_gen_block method creates a sequential model for a generator block with the given parameters.

The forward method passes the input noise through the generator architecture, producing an image.

The get_noise method generates random noise samples given the number of samples and the noise dimension.

**The Discriminator:**

The Discriminator class also inherits from nn.Module. It initializes the discriminator with the following parameters:

im_chan: The number of input channels (default is 3, for RGB images).
conv_dim: The dimension of the convolutional layers (default is 64).
image_size: The size of the input images (default is 64).
The discriminator architecture is a deep CNN consisting of a series of layers:

Five blocks of convolutional layers, each followed by batch normalization and LeakyReLU activation.
A final convolutional layer that produces a single scalar output representing the probability that the input image is real.
The make_disc_block method creates a sequential model for a discriminator block with the given parameters.

The forward method passes the input image through the discriminator architecture, producing a probability that the input image is real.

The _get_final_feature_dimention method calculates the final feature dimension of the discriminator. This method is not used in the current implementation, but could be useful for extracting features or calculating the output size of the final layer.

Loss calculation:

def real_loss(D_out, device='cpu'):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    batch_size = D_out.shape[0]
    labels = tf.ones((batch_size, 1), dtype=tf.float32) * 0.9  # real labels = 1 and label smoothing => 0.9

    loss = criterion(labels, D_out)
    return loss

def fake_loss(D_out, device='cpu'):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    batch_size = D_out.shape[0]
    labels = tf.zeros((batch_size, 1), dtype=tf.float32)  # fake labels = 0

    loss = criterion(labels, D_out)
    return loss


Both functions use Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) as the loss function. BCEWithLogitsLoss combines a sigmoid activation and binary cross-entropy loss in a single function, making it numerically stable and more efficient.
Hyper parameters:

z_dim = 150
beta_1 = 0.5
beta_2 = 0.999 

n_epochs = 150
lr = 0.0002

batch_size = 128
image_size = 128

Epoch (10)
![image](https://user-images.githubusercontent.com/40152283/235616334-86d34833-1ae7-4963-8136-4abea8597bec.png)
Epoch (50)
![image](https://user-images.githubusercontent.com/40152283/235616368-746ee339-557e-46ee-b723-c830909666cf.png)
Epoch (100)
![image](https://user-images.githubusercontent.com/40152283/235616428-d2566520-333e-430d-95ce-f71df8ae0d4a.png)

This pytorch code create abstract but clear images, a tint of uncanniness to the art style. But it is expected as we are combining all types of art into a single generation.\

Now we will see the difference if the same is generated using tensorflow.

Hyper parameters:
same, but learning rate of generator is 0.0008
![image](https://user-images.githubusercontent.com/40152283/235616857-d4d8205e-aa20-4b51-9628-c9cd5713cb4f.png)

![image](https://user-images.githubusercontent.com/40152283/235616895-5d2c414f-38c7-4f1b-be8e-a37be01ba93c.png)

![image](https://user-images.githubusercontent.com/40152283/235617018-61d96055-660d-4b98-ad67-764dcc53f980.png)

It seems the learning rate has severly altered the images.

We will see the difference when the learning rates are the same:
![image](https://user-images.githubusercontent.com/40152283/235617212-9cf1a527-48cc-4567-944b-a525a1b8caf3.png)
![image](https://user-images.githubusercontent.com/40152283/235617242-fd979168-55f9-412d-89cc-273bd8bbbf30.png)
![image](https://user-images.githubusercontent.com/40152283/235617263-72c95b5e-027d-4a0b-90d1-249bcc407dd7.png)

Better, alot of abstract, but has a more vibrant color and less uncanniness to the pytorch version. But I realized the generator loss is significantly higher than the discriminator.
Let's see what happens if we double the generator learning rate.

Generator learning rate = 0.0004

![image](https://user-images.githubusercontent.com/40152283/235617521-83bcedb0-0295-463c-b7e0-eab68fad06cf.png)
![image](https://user-images.githubusercontent.com/40152283/235617554-38fcd915-ff3b-47c7-b4f7-2babf8efac6e.png)
![image](https://user-images.githubusercontent.com/40152283/235617602-d1459e6f-78f7-44de-b1af-1e11f50c05b1.png)

Again, mostly very vague and abstract images.

Spectral normalization and dropout layers.
Spectral normalization to stablize the training process by normalizing th weights.
Dropout layer to prevent overfitting.
![image](https://user-images.githubusercontent.com/40152283/235618310-238c7177-94ac-435c-8d13-d348e3728723.png)
![image](https://user-images.githubusercontent.com/40152283/235618342-b4260f71-2c86-435a-a1e0-cbec8fc20741.png)
![image](https://user-images.githubusercontent.com/40152283/235618369-f26590ca-7887-4ddc-a948-36a5ff2f363c.png)

We are starting to see some of the clear but abstract images beginning to form. Perhaps longer epoch would beging creating some unique images.

**WGAN**
We implement wassersteing loss to see if there are any major differences with gradient penalty.

Wasserstein GAN (WGAN) improves GAN training by using the Wasserstein distance as the loss function, providing better convergence and stability. Gradient Penalty (GP) is a regularization technique for WGAN that penalizes the gradients of the critic function to ensure Lipschitz continuity, replacing weight clipping. Both WGAN and GP lead to better training stability and higher-quality generated samples.

Gradient Penalty (GP) is a regularization technique for Wasserstein GANs that ensures the Lipschitz continuity of the critic function. It replaces weight clipping by penalizing the gradients of the critic with respect to its inputs. The GP encourages the critic's gradients to have a norm close to 1, improving training stability and convergence without limiting the expressiveness of the critic function.

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pre
    
def real_loss(D_out):
    return wasserstein_loss(tf.ones_like(D_out), D_out)

def fake_loss(D_out):
    return wasserstein_loss(-tf.ones_like(D_out), D_out)
    
def gradient_penalty(real_images, fake_images, discriminator):
    batch_size = real_images.shape[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        pred = discriminator(interpolated_images)

    gradients = tape.gradient(pred, interpolated_images)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    return penalty


Immediately we see some major loss fluctuations and changes, but starts to stablize short later.
Epoch [    1/  100] | d_loss: -1063.5190 | g_loss: 609.3383
Epoch [    2/  100] | d_loss: -494.2930 | g_loss: -12887.6836
Epoch [    3/  100] | d_loss: 3389.8760 | g_loss: -2827.3130
Epoch [    4/  100] | d_loss: -130.8047 | g_loss: 5899.9795
Epoch [    5/  100] | d_loss: -2596.2517 | g_loss: -980.1146
.
.
.
.
Epoch [   46/  100] | d_loss: -0.5023 | g_loss: 1.4820
Epoch [   47/  100] | d_loss: 2.2713 | g_loss: -1.0798
Epoch [   48/  100] | d_loss: -0.1702 | g_loss: -0.2343
Epoch [   49/  100] | d_loss: -0.5616 | g_loss: -3.1981
Epoch [   50/  100] | d_loss: -0.1993 | g_loss: 2.5849

![image](https://user-images.githubusercontent.com/40152283/235619569-919a0fcf-d3e0-4625-99bd-18ef6f650cd7.png)
![image](https://user-images.githubusercontent.com/40152283/235619588-07ef7168-80b1-43fb-b2e4-dbe748fc40b2.png)
![image](https://user-images.githubusercontent.com/40152283/235619629-c9311cef-5e94-49f6-b1c2-8e1be70181f8.png)
Something happens ehre, the generated images begins to be worse with each iteration with more fluctuations
Epoch [   81/  100] | d_loss: -1.5930 | g_loss: -20.5108
Epoch [   82/  100] | d_loss: -1.9652 | g_loss: -12.6182
Epoch [   83/  100] | d_loss: -1.6951 | g_loss: -37.8707
Epoch [   84/  100] | d_loss: -11.8659 | g_loss: 49.9413
Epoch [   85/  100] | d_loss: -15.2518 | g_loss: 22.8540
![image](https://user-images.githubusercontent.com/40152283/235619660-3cc96af8-7844-434a-9b1a-d76794bceb61.png)
![image](https://user-images.githubusercontent.com/40152283/235619700-160ce126-5234-40ce-b20f-682573a935a2.png)

I suspect some overfitting, perhaps regularization, noise injection and learning rate scheduling or early stopping is necessary as the images around the middle were significantly better than the later ones.
result:
![image](https://user-images.githubusercontent.com/40152283/235621734-a616955a-90e4-4ce3-9794-3b2ce85aecba.png)
![image](https://user-images.githubusercontent.com/40152283/235621764-6f1b7665-b301-4f64-a112-3542fb4f5ca6.png)
![image](https://user-images.githubusercontent.com/40152283/235621780-60ec7e30-b4d2-41cf-abaa-21b72f73bf8e.png)
![image](https://user-images.githubusercontent.com/40152283/235621798-402a2990-f4d8-4055-8a47-88f5d9b0d7ae.png)


![image](https://user-images.githubusercontent.com/40152283/235621847-bfe33eb7-2d41-4d3e-ac46-90a74c1c68fe.png)


Next challenge progan.

Inorder to use progan, I need to maybe change environment from google colab (free version to paid) or a different framework entirely as there are not enough resources offered.


100%|██████████| 543/543 [05:06<00:00,  1.77it/s, gp=0.0232, loss_critic=-4.81]
=> Saving checkpoint
=> Saving checkpoint
Current image size: 256
Epoch [1/30]
100%|██████████| 543/543 [10:16<00:00,  1.14s/it, gp=0.0152, loss_critic=-6.79]
=> Saving checkpoint
=> Saving checkpoint
Epoch [2/30]
100%|██████████| 543/543 [10:17<00:00,  1.14s/it, gp=0.0309, loss_critic=-15.8]
=> Saving checkpoint
=> Saving checkpoint
Epoch [3/30]
100%|██████████| 543/543 [10:17<00:00,  1.14s/it, gp=0.0981, loss_critic=2.33]
=> Saving checkpoint
=> Saving checkpoint
Epoch [4/30]
100%|██████████| 543/543 [10:17<00:00,  1.14s/it, gp=0.264, loss_critic=11.7]
=> Saving checkpoint
=> Saving checkpoint
Epoch [5/30]
100%|██████████| 543/543 [10:17<00:00,  1.14s/it, gp=0.0352, loss_critic=0.744]
=> Saving checkpoint
=> Saving checkpoint
Epoch [6/30]
100%|██████████| 543/543 [10:17<00:00,  1.14s/it, gp=0.0248, loss_critic=-13.1]
=> Saving checkpoint
=> Saving checkpoint
Epoch [7/30]
 11%|█         | 58/543 [01:07<09:13,  1.14s/it, gp=0.0759, loss_critic=-7.29]






