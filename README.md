# 3D-CT-images-augmentation

# Training model
If you want to run training script for the first time, set RESUME_TRAINING flag to False.

If you want to continue training, set RESUME_TRAINING to True and set run_name to the name of the folder with model's state

# Launching tensorboard
tensorboard --port=8008 --logdir=$WGAN_SAVE_PATH/logs/29-11-2023_18:38
tensorboard --port=8009 --logdir=$WGAN_SAVE_PATH/logs/01-12-2023_23:55
tensorboard --port=8010 --logdir=$WGAN_SAVE_PATH/logs/03-12-2023_13:54


