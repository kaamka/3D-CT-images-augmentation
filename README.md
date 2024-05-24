# 3D-CT-images-augmentation

# Training model - 3D_CT_images_augmentation_wgan.py

Before running the script for the first time:
- set save_path where to save the model
- inside the save_path directory create two directories: logs/ and models/
- set directory variable to path to the dataset

If training model for the first time:
Set RESUME_TRAINING to False, run_name will be set based on current timestamp and used to save logs/models in the save_path directory

If continuing training:
Set RESUME_TRAINING to True
Set run_name to the timestamp of the original training
Set num_epochs_list, critic_iterations_list and generator_iterations_list

# Sampling model - load_model.py

1) Set checkpoint path
2) Paste corresponding implementation of the generator model

# Launching tensorboard
tensorboard --port=8008 --logdir=<save_path>/logs/<training_timestamp>


