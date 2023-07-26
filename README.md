# 3D-CT-images-augmentation

# Training model
If you want to run training script for the first time, set RESUME_TRAINING flag to False.

If you want to continue training, set RESUME_TRAINING to True and set run_name to the name of the folder with model's state

# Launching tensorboard
tensorboard --port=8009 --logdir=$WGAN_SAVE_PATH/logs/11-05-2023_14:25
tensorboard --port=8009 --logdir=$WGAN_SAVE_PATH/logs/23-05-2023_09:42
tensorboard --port=8009 --logdir=$WGAN_SAVE_PATH/logs/13-06-2023_11:51
tensorboard --port=8009 --logdir=$WGAN_SAVE_PATH/logs/24-07-2023_17:44
tensorboard --port=8009 --logdir=$WGAN_SAVE_PATH/logs/25-07-2023_12:36
