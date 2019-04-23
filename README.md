# NTU_actionrecognition
Tools for processing the NTU dataset using only RGB videos.

The dataset can be downloaded from here: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp

Video samples have been captured by 3 Microsoft Kinect v.2 cameras concurrently. The resolution of RGB videos is 1920×1080.

There are two two types of action classification evaluation: 

### Cross-Subject Evaluation

20 subjects for training and 20 different subjects for testing. The IDs of training subjects in this evaluation are: 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38; remaining subjects are reserved for testing. For this evaluation, the training and testing sets have 40, 320 and 16, 560 samples, respectively.

### Cross-View Evaluation

Samples of camera 1 for testing and samples of cameras 2 and 3 for training. For this evaluation, the training and testing sets have 37, 920 and 18, 960 samples, respectively.

### Read_data.py

Reads the addresses of the video and creates training and validation generators that can be used into a fit function.

### VideoGenerator.py

Datagenerator for keras.

