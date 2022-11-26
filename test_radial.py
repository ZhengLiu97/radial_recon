import scipy.io as sio
import numpy as np
from pynufft import NUFFT
import matplotlib.pyplot as plt


def combine_channels(image_data):
    return np.sqrt(np.sum(np.square(np.abs(image_data)), axis=2))


data = sio.loadmat('data/brain_64spokes.mat')
raw_data = data['rawdata']
raw_data_reshape = raw_data.reshape(-1, raw_data.shape[-1])
k = data['k']
print(k.shape, raw_data.shape, raw_data_reshape.shape)
M = raw_data.shape[0] * raw_data.shape[1]

# Create a M*2 np array
# To store the coordinates of k-space points

# om = np.empty((M, 2))
# om[:, 0] = k.real.flatten() * 2 * np.pi  # Normalize the k-space to (-pi, pi)
# om[:, 1] = k.imag.flatten() * 2 * np.pi

# a more generalized method
om = np.empty((M, 2))
om[:, 0] = k.real.flatten()
om[:, 1] = k.imag.flatten()
om = (om - om.min(axis=0)) / (om.max(axis=0) - om.min(axis=0)) * 2 * np.pi - np.pi

# image size
Nd = (256, 256)
print('setting image dimension Nd...', Nd)
# k-space size
Kd = (512, 512)
print('setting spectrum dimension Kd...', Kd)
# interpolation size
Jd = (6, 6)
print('setting interpolation size Jd...', Jd)

NufftObj = NUFFT()
NufftObj.plan(om, Nd, Kd, Jd)

img = np.empty((Nd[0], Nd[1], raw_data_reshape.shape[1]), dtype=np.complex128)
for index, current_raw_data in enumerate(raw_data_reshape.transpose(1, 0)):
    img[:, :, index] = NufftObj.solve(current_raw_data, solver='cg', maxiter=50)
image_final = combine_channels(img)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].plot(om[:, 0], om[:, 1], 'ko', markersize=1)
axes[0].set_aspect(1)
axes[0].set_title('k-space radial trajectory ')
axes[0].axis('off')

axes[1].imshow(image_final, cmap='gray')
axes[1].set_aspect(image_final.shape[0] / image_final.shape[1])
axes[1].set_title('image')
axes[1].axis('off')
