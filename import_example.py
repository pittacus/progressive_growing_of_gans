import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

N=10

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
# with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
with open('oct2014-network-snapshot-006012.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate latent vectors.
# latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
# latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

latents = np.random.RandomState(19810418).randn(N, *Gs.input_shapes[0][1:]) # 1000 random latents

latents += np.random.randn(N, *Gs.input_shapes[0][1:])*0.3
# print(latents)
# latents[0][0]=512
# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

print("# Run the generator to produce a set of images.")
images = Gs.run(latents, labels)
print(images.shape)

print("# Convert images to PIL-compatible format.")
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

print("# Save images as PNG.")
for idx in range(images.shape[0]):
    if images.shape[-1]==1:
        PIL.Image.fromarray(np.squeeze(images[idx],axis=2), 'L').save('img%d.png' % idx)
    elif images.shape[-1]==3:
        PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)
    else:
        print("# Save images failed: unknown channel count.", images[idx].shape)

print("# Save images success.")
