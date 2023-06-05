# iml

## Introduction
Image retrieval is a crucial machine learning task, that is to say, a specific problem or objective that the model is designed to solve. It aims to search for images in a dataset, similar to the features of a given query image.  Its importance is directly connected to the relevance that images have, for the human species, to convey information and thus it is a highly researched field. (Halawani et al., 2006)
A common technique for image retrieval is “text-based image retrieval” (Liu et al., 2007). However, it is not efficient as it forces the programmer to annotate by hand, in just one language,  big databases. Another method is “content-based image retrieval” (CBIR), which relies instead on matching features from the query image to the ones of the given database. The latter is recommended by literature, as more straightforward and reliable (Lu and Weng, 2007). 
CBIR utilizes feature extraction techniques to collect relevant information about the image ( such as color, edge features,  texture features, etc.) and compare it through similarity measures ( such as Euclidean distance, Cosine similarity, Jaccard Similarity, etc.)  to a database of possible matches.
Although it might sound similar to other tasks such as image classification, or object detection, it is a unique approach. In contrast to classification methods that aim to assign labels or categories to images, this task focuses on finding similarities with a query image, given as an input. Object detection, instead, involves using different algorithms and models and attempts to identify and localize specific objects of interest within an image, providing detailed information about their presence and spatial location. (Wan et al., 2014)

This report outlines popular CBIR techniques within Machine Learning to maximize the accuracy with which it is asked to match a picture of a human face, with a provided dataset (Figure 1). The project has been coded completely in Python3.

Figure 1: A Black Box View of the Task


## Methodology

The Dataset
To improve Machine Learning results and prevent overfitting, data augmentation is crucial. It involves modifying the existing data to expand the training set. We use basic augmentations such as reshaping, RGB transformation, and normalization. The images are then converted into tensor data before being fed into the model.

To enhance the results’ accuracy, a dataset consisting of more than 200,000 images and over 10,000 labels was utilized (CelebA, www.kaggle.com, n.d.). We opted to sample the dataset, retaining only the images from the 481 most frequent classes. Each class had to have a minimum of 29 images to be included, to prevent class imbalances within the dataset. 
The photos were stored in a folder as PNG files, with size 224x224, and the corresponding labels were saved in a text document. The data were randomly split into train and validation sets using indexes, with 80% used for training and 20% for validation.
We faced a face-detection challenge during the upload process. To streamline the code and reduce computational overhead, we used the Haarcascade algorithm instead of developing a Faster R-CNN model. This decision resulted in a lighter solution.

Model 

Metric Learning 
Distance Metric Learning can be seen as a preliminary stage in distance-based learning algorithms (Suárez, García, and Herrera, 2021.). In our specific task, we have implemented a “Triplet Loss Network”, due to its ability to learn similarities directly from the data and the embedding-based approach, which allows for efficient and scalable search (Schroff, Kalenichenko, and Philbin, 2015).
A Triplet Loss Network comprises three sub-networks with identical architecture and weights (Hoffer and Ailon, 2018).  Each sub-network takes a single input and produces a feature representation as its output, which is then used to learn the similarity and dissimilarity between inputs (Figure 2).

For training, backpropagation is used through the Triplet Loss function:

The loss function encourages the anchor and positive examples to be closer to each other than the anchor and negative examples, by at least a specified margin value. This margin acts as a threshold, ensuring that the positive and negative pairs are sufficiently separated in the embedding space. By minimizing the triplet loss, the network learns to produce embeddings that effectively discriminate between different classes or categories.


Transfer Learning
In the “Triplet Loss Network” architecture, each subnetwork has been constructed using Transfer Learning techniques. Specifically, for the task of image retrieval and comparison, three "twin" networks have been employed, all of which are pre-trained ResNet50 models. These pre-trained models serve as the basis for extracting and defining the features required to compare query images and have been chosen due to their great discriminative feature learning techniques (Figure 3).

Figure 3: ResNet50 Architecture (He et al., 2015)



