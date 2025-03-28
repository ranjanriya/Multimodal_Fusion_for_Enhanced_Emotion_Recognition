# A comparison of shared encoders for multimodal emotion recognition
## Course Project, CSCI 535, Spring 2024
Contributors - Sai Anuroop Kesanapalli, Riya Ranjan, Aashi Goyal, Wilson Tan

#### For 2D experiments

* Generate mel spectrograms from WAV audio files by running ```wav_to_melspec.py```<br>
  ```python3 wav_to_melspec.py /path/to/WAVFiles /path/to/output```

* Run the notebooks corresponding to `[ResNet18, GoogLeNet, VGG16, ViT, PTViT]` in `/2D` to train the respective encoders on `[audio, vision, multimodal]` data. 

> **NOTE:** Pre-processed data (faces and spectrograms) is stored as ```.npy``` files in [GDrive1](https://drive.google.com/drive/folders/1BhpgUDgbYwoTaTO6Yo8M3uR0Clw0bkiC?usp=sharing) and [GDrive2](https://drive.google.com/drive/folders/1Q1LFiq2KZPyYTuEJhbQY38uu9FE0Jl-g?usp=sharing). Grayscale and RGB data is used for ```[ResNet18, ViT]``` and ```[GoogLeNet, VGG16, PTViT]``` respectively.

#### For 3D experiments

* Start with creating mel spectrograms by running:

  ```python3 wav_to_melspec_3d.py /path/to/input_folder /path/to/output_folder```

* Then create 3D Data:

  ```python3 create_3d_data.py /path/to/video_folder /path/to/spectrogram_folder /path/to/output_folder```

* Now train your model with any of the following where ```modality``` can be ```[audio, vision, multi]```. ```/path/to/pretrain_checkpoint``` is optional. The ImageNet pretrained model used here is provided in the ```models``` folder at [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) named ```rgb_imagenet.pt```. If ```/path/to/pretrain_checkpoint``` is missing, the untrained I3D model will be used:

  ```python3 simple3d_train_test.py modality /path/to/3d_data /path/to/output_folder```

  ```python3 i3d_train_test.py modality /path/to/3d_data /path/to/output /path/to/pretrain_checkpoint```

  ```python3 videoMAE_train_test.py modality /path/to/3d_data /path/to/output_folder```

* For ablated tests. ```/path/to/checkpoint``` is optional but recommended, otherwise an untrained model is used:

  ```python3 simple3d_ablated_test.py modality /path/to/3d_data /path/to/checkpoint```

  ```python3 i3d_ablated_test.py modality /path/to/3d_data /path/to/checkpoint```
