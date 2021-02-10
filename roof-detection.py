# This Python script was adapted from https://github.com/karolzak/keras-unet/blob/master/notebooks/kz-whale-tails.ipynb

from prepare_and_save_data import *
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
from keras_unet.models import satellite_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded, dice_coef
from keras_unet.utils import plot_segm_history
from keras_unet.utils import plot_imgs
# from loss_functions import f1_loss

##############
# PARAMETERS #
##############

# Parameters for data augmentation
batch_size = 8
rotation_range = 10.
channel_shift_range = 0.1
zoom_range = 0.2

# Training hyperparameters
steps_per_epoch = 200
epochs = 15
val_ratio = 0 # Full training set

# Model file names
model_filename_base = 'BCE_loss'
model_filename_best = model_filename_base + '-Best.h5'
model_filename_final = model_filename_base + '-Final.h5'

# Directories
directory = r'data'
test_directory = r'data/test/'
predictions_directory = r'results/BCE_loss/predictions/'

####################
# DATA PREPARATION #
####################

images, labels = load_data(directory)
image_size = images.shape[1]

print("Shape of Input: " + str(images.shape))
print("Shape of Labels: " + str(labels.shape))

# Train/val split
images_train = images
labels_train = labels
images_val = None
labels_val = None

if val_ratio != 0:
    images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=val_ratio, random_state=0)

print("images_train: ", images_train.shape)
print("labels_train: ", labels_train.shape)

if val_ratio != 0:
    print("images_val: ", images_val.shape)
    print("labels_val: ", labels_val.shape)

# Prepare train generator with data augmentation
train_gen = get_augmented(
        images_train, labels_train, batch_size=batch_size,
        data_gen_args=dict(
            rotation_range=rotation_range,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0,
            channel_shift_range=channel_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant'
        ))

###########################
# SETUP AND TRAIN NETWORK #
###########################

# Initialize network
input_shape = images_train[0].shape
model = satellite_unet(input_shape)
model.summary()

# Compile & train
if val_ratio != 0:
    callback_checkpoint = ModelCheckpoint(
        model_filename_best,
        verbose=1,
        monitor='val_loss',
        save_best_only=True
    )
else:
    callback_checkpoint = ModelCheckpoint(
        model_filename_best,
        verbose=1,
        monitor='loss',
        save_best_only=True
    )

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    # loss=f1_loss,
    metrics=[iou, iou_thresholded, dice_coef]
)

if val_ratio != 0:
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(images_val, labels_val),
        callbacks=[callback_checkpoint]
    )
else:
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[callback_checkpoint]
    )

model.save_weights(model_filename_final)

# Plot training history
if val_ratio != 0:
    plot_segm_history(history)
else:
    plot_segm_history(history, metrics=["iou"], losses=["loss"])

####################
# PLOT TEST RESULT #
####################

# Plot original + pred + overlay (pred on top of original)
images_test, file_names = load_test_data(test_directory)

# Predictions from best model according to val_loss
if val_ratio != 0:
    model.load_weights(model_filename_best)
    labels_pred_best = model.predict(images_test)
    plot_imgs(org_imgs=images_test, mask_imgs=labels_pred_best, nm_img_to_plot=10)

# Predictions from final model
model.load_weights(model_filename_final)
labels_pred_final = model.predict(images_test)
plot_imgs(org_imgs=images_test, mask_imgs=labels_pred_final, nm_img_to_plot=10)

# Save final predictions
save_test_predictions(labels_pred_final, predictions_directory, file_names)
