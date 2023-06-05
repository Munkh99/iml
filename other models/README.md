# Introduction to machine learning competition project 2023
## General info
In this project we will create an image search engine where a query image is fed to one of the three models (ResNet50, VGG16 and EfficientNet) and will return the most N similar images from a gallery.
## Pre-requisites
- Python 3.10.10
## Requirements
The `requirements.txt` file contains a list of dependencies required to run this project. These dependencies are external packages or libraries that need to be installed in order for the project to function properly.

## Installation

To install the required dependencies, you can use the following command:

```shell
pip install -r requirements.txt
```
## Run
1. Set the values for the variables and paths on the `config/config.yaml` file.
2. Run the `tran.py` file with selected --model_name (name of the model), --checkpoint_path (path to save checkpoints) and --config_path (path to config file) to train the selected models. After training a copy of the config file, checkpoint file and a log file will be created.
3. Run the `test.py` file with selected --model_name and --checkpoint_path, which will load the best state of the model achieved in training and run it on the test data set and sumbit the top 10 most similiar images to the url provided in file. After process has finished an accuracy result for the top 1, top 5 and top 10 images will be recieved in response.
