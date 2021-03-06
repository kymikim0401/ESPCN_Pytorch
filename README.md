# ESPCN from scrath with custom dataset
# [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158.pdf)
![68747470733a2f2f692e696d6775722e636f6d2f54315a584c4d302e706e67](https://user-images.githubusercontent.com/82307352/168478374-5175a83a-54f3-40e1-ad01-a95e644fd51a.png)


## Introduction
ESPCN directly uses low-resolution (LR) image as an input and is upscaled at the very last stage of the architecture through "sub-pixel" convolution layer. Basically, it is an interpolation of LR image with trained, previous feature maps which is called "sub-pixel CNN" **(See above figure for the better intuition)**. Unlike SRCNN, ESPCN does not require the LR image to be upsampled before putting it into the input layer. In terms of computation complexity and memory cost, it is beneficial since the model does not need to apply the CNN directly to the upsampled LR image. 

## Dataset
Custom training dataset was obtained from the below link from kaggle:
https://www.kaggle.com/datasets/saputrahas/dataset-image-super-resolution
It has two data folders: 685 images of training set and 170 images of validation set. Due to lack of training and validation set for a proper model training, and due to **dataset subsampling**(the author uses 91-images dataset and run data augmentation to make sub-images for the training set) mentioned in the paper, I made *Data_augmentation.py*, which is random cropping data augmentation code. This enabled me to train the model with 6850 sub images and validate with 1700 sub-images. With this data-augmented training set, I made LR images with upscale factor = 3 and bicubic downsampling. 

## Train
For the training, I used Adam optimizer with the learning rate = 1e-4. For the activation function, I tried both tanh and relu for the comparison. I used both MSE loss and PSNR (peak signal-to-noise ratio) for performance metric. PSNR naively represents how similar the model output is compared to the ground truth, measured in dB scale. 

## Test 
After the training, I made test_video.py, which basically applies super resolution to low-resoltuion video. Opencv library provides "*VideoCapture*" and "*VideoWriter*", which basically decomposes an input video into individual frames and save video files with the trained ESPCN model output of those frames. 

## Results
With 20 Epoch for the training, below is the MSE error and PSNR with respect to each epoch, for both training and validation. With my custom data, ESPCN with ReLU activation layer gives a better PSNR than Tanh layer. 
Below is the result with ReLU activation layer:
| Activation     | PSNR (dB) |
| ---  | :---:  |
| ReLU | 23.579 |
| Tanh | 23.332 |

![relu](https://user-images.githubusercontent.com/82307352/168479217-2dcfa068-f040-4f37-9b76-0b3023798d9f.png)

Below is the result with tanh activation layer:
![tanh](https://user-images.githubusercontent.com/82307352/168479287-818dc324-772a-47f1-97e7-5e549d95c19c.jpg)

For video test, I used a low-resolution, concert live video from one of my favorite singers in South Korea, Moon Sae Lee. Here is the Youtube link of the video: https://www.youtube.com/watch?v=t1qq60p8rz8

Below is the comparison between original youtube video and result video with our custom - trained ESPCN:
| Result | Original |
| :---:  | :---:  |
|![test_result](https://user-images.githubusercontent.com/82307352/168539455-42904b21-2fd4-462e-ab0c-7efe63b14616.gif)|![test_](https://user-images.githubusercontent.com/82307352/168539674-147e64e2-5a71-4063-8952-f133903adae7.gif)|

It is bit unclear to notice the improvements from above sample since only Y channel has been trained. In order to fully apply the super resolution, all 3 RGB channels should be trained and used. However, there are some frames that seemed to be improved with ESPCN:
![??????1](https://user-images.githubusercontent.com/82307352/168540595-5c8db63c-f1be-4441-967e-d6e3ba3f615d.jpg)


Here, we can see that: while the original video seems to be more gritty, rough, and unnecessarily glossy, the result video is somewhat smooth, vivid, seemed to be more natural. 




