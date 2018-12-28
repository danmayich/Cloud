# Cloud Segmentation
Researching if Tensorflow along with Deeplab is able to properly segment cumulus clouds to determine actual cloud coverage.<br>

The end goal of the project is to get a highly accurate masked prediction on cloud images.

![alt text](https://raw.githubusercontent.com/danmayich/Cloud/master/trainingdata/images/21600_000.png "Cloud")
![alt text](https://raw.githubusercontent.com/danmayich/Cloud/master/trainingdata/maskedimages/21600_000.png "Masked")
<br>
We are starting this project using generated data from a large eddy simulation, eventually hoping to move onto real cloud images from a Total Sky Imager.<br>

This read-me will describe how to set up and run the cloud segmentation project.

## Setup
To use this project you must first install Tensorflow and Deeplab.<br>
[Deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)<br>
[Tensorflow](https://www.tensorflow.org/)<br>
<br>
If everything is already installed you can jump directly to the Running section.

## Running
Run these scripts in the following order:
1) generate_records.py <br>Generates the tensorflow records.<br>
2) train.py <br>Trains the model.<br>
3) get_results.py <br>Generates the results.

## Datasets
This repo includes a small dataset of masked and non-masked cloud images. Which can be expanded upon/modified along with the list files to change the datasets used for training.
<br>(See ./trainingdata section.)

## Folders
Description of the folders and what they should contain.

### ./checkpoints
Checkpoints will be saved to this location.
<br>

### ./results
Segmentation outputs will be saved to the following sub-folders.<br>
1) ./segmentation_results<br> Labeled segmentation results (Color).
2) ./raw_segmentation_results <br>Unlabeled segmentation results (B/W).

### ./tfrecords
The generated Tensorflow records will be saved to this location.

### ./trainingdata
1) ./images<br> Images used for training and validation.

2) ./maskedimages <br> Pre-masked images used for training and validation.

* train.txt<br> List of images used for training.<br>
* trainval.txt<br> List of images used for training validation.<br>
* val.txt<br> List of images used for validation.<br>
