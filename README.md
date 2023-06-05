# iml

## Introduction
Image retrieval is a crucial machine learning task, that is to say, a specific problem or objective that the model is designed to solve. It aims to search for images in a dataset, similar to the features of a given query image.  Its importance is directly connected to the relevance that images have, for the human species, to convey information and thus it is a highly researched field. (Halawani et al., 2006)
A common technique for image retrieval is “text-based image retrieval” (Liu et al., 2007). However, it is not efficient as it forces the programmer to annotate by hand, in just one language,  big databases. Another method is “content-based image retrieval” (CBIR), which relies instead on matching features from the query image to the ones of the given database. The latter is recommended by literature, as more straightforward and reliable (Lu and Weng, 2007). 
CBIR utilizes feature extraction techniques to collect relevant information about the image ( such as color, edge features,  texture features, etc.) and compare it through similarity measures ( such as Euclidean distance, Cosine similarity, Jaccard Similarity, etc.)  to a database of possible matches.
Although it might sound similar to other tasks such as image classification, or object detection, it is a unique approach. In contrast to classification methods that aim to assign labels or categories to images, this task focuses on finding similarities with a query image, given as an input. Object detection, instead, involves using different algorithms and models and attempts to identify and localize specific objects of interest within an image, providing detailed information about their presence and spatial location. (Wan et al., 2014)

This report outlines popular CBIR techniques within Machine Learning to maximize the accuracy with which it is asked to match a picture of a human face, with a provided dataset (Figure 1). The project has been coded completely in Python3.
![alt text](https://github.com/Munkh99/iml/blob/master/figures/Screenshot%202023-06-05%20at%2010.50.55.png)
Figure 1: A Black Box View of the Task


## Methodology

### The Dataset
To improve Machine Learning results and prevent overfitting, data augmentation is crucial. It involves modifying the existing data to expand the training set. We use basic augmentations such as reshaping, RGB transformation, and normalization. The images are then converted into tensor data before being fed into the model.

To enhance the results’ accuracy, a dataset consisting of more than 200,000 images and over 10,000 labels was utilized (CelebA, www.kaggle.com, n.d.). We opted to sample the dataset, retaining only the images from the 481 most frequent classes. Each class had to have a minimum of 29 images to be included, to prevent class imbalances within the dataset. 
The photos were stored in a folder as PNG files, with size 224x224, and the corresponding labels were saved in a text document. The data were randomly split into train and validation sets using indexes, with 80% used for training and 20% for validation.
We faced a face-detection challenge during the upload process. To streamline the code and reduce computational overhead, we used the Haarcascade algorithm instead of developing a Faster R-CNN model. This decision resulted in a lighter solution.

### Model 

#### Metric Learning 
Distance Metric Learning can be seen as a preliminary stage in distance-based learning algorithms (Suárez, García, and Herrera, 2021.). In our specific task, we have implemented a “Triplet Loss Network”, due to its ability to learn similarities directly from the data and the embedding-based approach, which allows for efficient and scalable search (Schroff, Kalenichenko, and Philbin, 2015).
A Triplet Loss Network comprises three sub-networks with identical architecture and weights (Hoffer and Ailon, 2018).  Each sub-network takes a single input and produces a feature representation as its output, which is then used to learn the similarity and dissimilarity between inputs (Figure 2).
![alt text](https://github.com/Munkh99/iml/blob/master/figures/Screenshot%202023-06-05%20at%2010.51.22.png)
Figure 2: The Triplet Loss Network Architecture


For training, backpropagation is used through the Triplet Loss function:
![alt text](https://github.com/Munkh99/iml/blob/add7ba5bf8c96e691ad33bd617d1cb798cd29c99/figures/Screenshot%202023-06-05%20at%2011.03.27.png)
The loss function encourages the anchor and positive examples to be closer to each other than the anchor and negative examples, by at least a specified margin value. This margin acts as a threshold, ensuring that the positive and negative pairs are sufficiently separated in the embedding space. By minimizing the triplet loss, the network learns to produce embeddings that effectively discriminate between different classes or categories.


#### Transfer Learning
In the “Triplet Loss Network” architecture, each subnetwork has been constructed using Transfer Learning techniques. Specifically, for the task of image retrieval and comparison, three "twin" networks have been employed, all of which are pre-trained ResNet50 models. These pre-trained models serve as the basis for extracting and defining the features required to compare query images and have been chosen due to their great discriminative feature learning techniques (Figure 3).
![alt text](https://github.com/Munkh99/iml/blob/ba09d0c4eace11c4003c86d91664d37300ca80bf/figures/Screenshot%202023-06-05%20at%2010.51.37.png)
Figure 3: ResNet50 Architecture (He et al., 2015)

#### Inference Procedure
The inference process is performed by the `distance_estimator` function, which takes the query set and the gallery set as inputs. It compares the extracted features from each image to determine the distance or dissimilarity between them. This distance estimation helps quantify the similarity or discrepancy between the query images and the gallery images. By analyzing the feature distances, the function can provide insights into the similarity rankings or identify potential matches between the query and gallery images (Figure 4).
![alt text](https://github.com/Munkh99/iml/blob/add7ba5bf8c96e691ad33bd617d1cb798cd29c99/figures/Screenshot%202023-06-05%20at%2010.51.46.png)
Figure 4: Inference procedure

#### Other models 
For our project, we have explored a classification-driven approach incorporating three prmodels: ResNet50, EfficientNet, and VGG16. EfficientNet has shown superior performance compared to other networks in literature  (Tan and Le, 2019), thus the choice of including it in our evaluation. These models have trained on the sampled "CelebA" dataset, which underwent pre-processing and face cropping. For face detection in the testing phase, we utilized the Haarcascade algorithm. 

## Results
The results were saved in a JSON file containing our group name as a value, the key “image” connected to the query, and the retrieved results expressed as a list. The “Triplet Loss Network” performed as follows on the challenge day:
 
![alt text](https://github.com/Munkh99/iml/blob/6a48659ec3b6ccdd844aa95d7f417a4c59495148/figures/Screenshot%202023-06-05%20at%2011.07.56.png)
Table 1: Results from the Triplet Loss Network

The results differed significantly from what was observed during the training and validation procedure, as shown below:
![alt text](https://github.com/Munkh99/iml/blob/6a48659ec3b6ccdd844aa95d7f417a4c59495148/figures/Figure_2.png)
![alt text](https://github.com/Munkh99/iml/blob/6a48659ec3b6ccdd844aa95d7f417a4c59495148/figures/Figure_1.png)
Figure 5: Accuracy and Loss from our Dataset

The other implemented approaches yielded the following results:
[alt text](https://github.com/Munkh99/iml/blob/6a48659ec3b6ccdd844aa95d7f417a4c59495148/figures/Screenshot%202023-06-05%20at%2011.08.41.png)
Table 2: Results from the ResNet50, EfficientNet, VGG16

Among the models evaluated, ResNet50 emerged as the top-performing model for image retrieval. An example of the retrieval process is shown below:
![alt text](https://github.com/Munkh99/iml/blob/6a48659ec3b6ccdd844aa95d7f417a4c59495148/figures/Screenshot%202023-06-05%20at%2011.08.56.png)
Figure 6: Query Example

## Discussion
The triplet loss model is effective for image retrieval tasks, learning discriminative embeddings, and encouraging closer mapping of similar examples. Moreover, metric learning principles enable efficient searching for similar images based on embedding distances. To conclude, the model is very flexible and can easily be adapted to different types of tasks. However, computational costs are high, especially with large datasets, and selecting informative triplets is crucial for successful training, but can be challenging and time-consuming.
As mentioned before, other 3 models were tested to compare with our main solution: VGG16, Resnet50, and EfficientNet.  The ResNet50 model achieved the best results, due to its deeper architecture. Compared to VGG16 and EfficientNet, ResNet50's increased depth allows it to learn more complex hierarchical features more easily, contributing to its superior performance. Moreover, it is more robust to noise and variations, a fundamental characteristic when dealing with high-volume queries and galleries. It goes so far as to beat the performance of the triplet loss network, as better discriminative feature learning techniques may advantage it.

## Conclusions
Our project highlighted the efficiency and accuracy of the ResNet50 pre-trained model compared to other models we tested. It outperformed all other models in terms of performance and demonstrated superior computational manageability and speed. 
Despite the performance of ResNet50, we acknowledge that there is room for further improvement in our Triplet Loss Network. We believe that with enhancements, it has the potential to achieve, if not surpass, the accuracy of ResNet50. 
One aspect we considered is the possibility of our model being affected by a domain shift, which could potentially reduce its performance. To address this, we propose implementing a Discriminative Adversarial Neural Network. By incorporating a domain classifier, we aim to better define and handle the domain shift, potentially resulting in improved performance and robustness.
Furthermore, by carefully selecting datasets and paying more attention to augmentations and cropping techniques, we could have potentially improved the performance of our model. These considerations could lead to better generalization and robustness. In conclusion, it would be intriguing to explore the utilization of YOLOv8 for face detection. 

## Contributions
The research into different pieces of literature and possible datasets was done as a group effort, to begin working on the project.  
The Triplet Loss Network has been mostly redefined in the architecture by Munkhdelger, while Pooria and Alice have dealt with data augmentation and face-cropping. 
The alternative approaches have been implemented by Pooria and Alice, however, the main code structure was changed and cleaned by Pooria.
The report was written by Alice, and figures and diagrams were drawn by Pooria and  Munkhdelger.

## References

Halawani, A., Teynor, A., Brunner, G. and Burkhardt, H. (2006). Fundamentals and Applications of Image Retrieval: An Overview. Bioimaging and Analysis View project Ethics and Agile Software Development -Mapping the Landscape: A Mapping Review Protocol View project Fundamentals and Applications of Image Retrieval.
Hameed, I.M., Abdulhussain, S.H. and Mahmmod, B.M. (2021). Content-based image retrieval: A review of recent trends. Cogent Engineering, 8(1), p.1927469. doi:https://doi.org/10.1080/23311916.2021.1927469.
He, K., Zhang, X., Ren, S. and Sun, J. (2015). Deep Residual Learning for Image Recognition. [online] Available at: https://arxiv.org/pdf/1512.03385v1.pdf.
Hoffer, E. and Ailon, N. (2018). Deep metric learning using Triplet network. arXiv:1412.6622 [cs, stat]. [online] Available at: https://arxiv.org/abs/1412.6622.
Hosna, A., Merry, E., Gyalmo, J., Alom, Z., Aung, Z. and Azim, M.A. (2022). Transfer learning: a friendly introduction. Journal of Big Data, 9(1). doi:https://doi.org/10.1186/s40537-022-00652-w.
Liu, Y., Zhang, D., Lu, G. and Ma, W.-Y. (2007). A survey of content-based image retrieval with high-level semantics. Pattern Recognition, 40(1), pp.262–282. doi:https://doi.org/10.1016/j.patcog.2006.04.045.
Lu, D. and Weng, Q. (2007). A survey of image classification methods and techniques for improving classification performance. International Journal of Remote Sensing, [online] 28(5), pp.823–870. doi:https://doi.org/10.1080/01431160600746456.
Schroff, F., Kalenichenko, D. and Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [online] doi:https://doi.org/10.1109/cvpr.2015.7298682.
Suárez, J.L., García, S. and Herrera, F. (2021). A tutorial on distance metric learning: Mathematical foundations, algorithms, experimental analysis, prospects and challenges. Neurocomputing, 425, pp.300–322. doi:https://doi.org/10.1016/j.neucom.2020.08.017.
Tan, M. and Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [online] arXiv.org. Available at: https://arxiv.org/abs/1905.11946.
Wan, J., Hoi, S., Wu, P., Zhu, J., Wang, D., Wu, Zhang, Y. and Li (2014). Deep learning for content-based image retrieval: A comprehensive study. pp.157–166.
Wang, J., Song, Y., Leung, T., Rosenberg, C., Wang, J., Philbin, J., Chen, B. and Wu, Y. (2014). Learning Fine-grained Image Similarity with Deep Ranking. [online] Available at: https://arxiv.org/pdf/1404.4661.pdf [Accessed 2 Jun. 2023].
www.kaggle.com. (n.d.). CelebFaces Attributes (CelebA) Dataset. [online] Available at: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.






