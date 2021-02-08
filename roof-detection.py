from prepare_data import *
from keras_unet.utils import plot_imgs

# Parameters for data augmentation
batch_size = 10
rotation_range = 5.
channel_shift_range = 0.1
zoom_range = 0.2

# Directory and image size
imgs_np, masks_np = load_data(r'data/images/')
image_size = imgs_np.shape[1]

print("Shape of X: " + str(imgs_np.shape))
print("Shape of Y: " + str(masks_np.shape))

# Plot images + masks + overlay (mask over original)
# plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)

# Get data into correct shape, dtype and range (0.0-1.0)
x = np.asarray(imgs_np, dtype=np.float32)
y = np.asarray(masks_np, dtype=np.float32)

# Train/val split
x_train, x_val, y_train, y_val = train_val_split(x, y, 0.2)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

# Prepare train generator with data augmentation
train_gen = get_train_set_augmented(x_train, y_train, batch_size, rotation_range, channel_shift_range, zoom_range)

sample_batch = next(train_gen)
xx, yy = sample_batch
print(xx.shape, yy.shape)

plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=batch_size, figsize=6)