from prepare_data import *
from keras_unet.utils import plot_imgs
from keras_unet.models import satellite_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.utils import plot_segm_history
from keras_unet.utils import plot_imgs

#######################################################
#################### PARAMETERS #######################
#######################################################

# Parameters for data augmentation
batch_size = 8
rotation_range = 5.
channel_shift_range = 0.1
zoom_range = 0.2

# Training hyper paramters
steps_per_epoch = 200
epochs = 7

# Model file name
model_filename = 'segm_model_v3.h5'

# Directory
directory = r'data/images/'

#######################################################
################# DATA PREPARATION ####################
#######################################################

imgs_np, masks_np = load_data(directory)
image_size = imgs_np.shape[1]

print("Shape of X: " + str(imgs_np.shape))
print("Shape of Y: " + str(masks_np.shape))

# Plot images + masks + overlay (mask over original)
# plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)

# Get data into correct shape, dtype and range (0.0-1.0)
x = np.asarray(imgs_np, dtype=np.float32)
y = np.asarray(masks_np, dtype=np.float32)

# Train/val split
x_train, x_val, y_train, y_val = train_val_split(x, y, 0.25)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

# Prepare train generator with data augmentation
train_gen = get_train_set_augmented(x_train, y_train, batch_size, rotation_range, channel_shift_range, zoom_range)

sample_batch = next(train_gen)
xx, yy = sample_batch
print(xx.shape, yy.shape)

# plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=batch_size, figsize=6)

#######################################################
############## SETUP AND TRAIN NETWORK ################
#######################################################

# Initialize network
input_shape = x_train[0].shape
model = satellite_unet(input_shape)
model.summary()

# Compile & train
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)

model.compile(
    optimizer=Adam(),
    # optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    # loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)

# Plot training history
plot_segm_history(history)

# Plot original + pred + overlay (pred on top of original)
model.load_weights(model_filename)
y_pred = model.predict(x_val)

plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=10)