# Capybara: Introduction to Machine Learning project

Image retrieval is a crucial machine learning task, that is to say, a specific problem or objective that the model is designed to solve. It aims to search for images in a dataset, similar to the features of a given query image.  Its importance is directly connected to the relevance that images have, for the human species, to convey information and thus it is a highly researched field. 

A common technique for image retrieval is “text-based image retrieval” . However, it is not efficient as it forces the programmer to annotate by hand, in just one language,  big databases. Another method is “content-based image retrieval” (CBIR), which relies instead on matching features from the query image to the ones of the given database. The latter is recommended by literature, as more straightforward and reliable. 

CBIR utilizes feature extraction techniques to collect relevant information about the image ( such as color, edge features,  texture features, etc.) and compare it through similarity measures ( such as Euclidean distance, Cosine similarity, Jaccard Similarity, etc.)  to a database of possible matches.

Although it might sound similar to other tasks such as image classification, or object detection, it is a unique approach. In contrast to classification methods that aim to assign labels or categories to images, this task focuses on finding similarities with a query image, given as an input. Object detection, instead, involves using different algorithms and models and attempts to identify and localize specific objects of interest within an image, providing detailed information about their presence and spatial location. 

This report outlines popular CBIR techniques within Machine Learning to maximize the accuracy with which it is asked to match a picture of a human face, with a provided dataset (Figure 1). The project has been coded completely in Python3.
![alt text](https://github.com/Munkh99/iml/blob/master/figures/Screenshot%202023-06-05%20at%2010.50.55.png)
Figure 1: A Black Box View of the Task

## Pre-requisites

The necessary libraries to execute our code are:

- `pip install pandas`
- `pip install pandas`
- `pip install -U scikit-learn`
- `pip install Pillow`
- `pip install torch`
- `pip install torchvision`
- `pip install opencv-python`
- `pip install more-itertools`
- `pip install glob2`
- `pip install wandb`
- `pip install PyYAML`


## Architecture

#### Metric Learning 
Distance Metric Learning can be seen as a preliminary stage in distance-based learning algorithms (Suárez, García, and Herrera, 2021.). In our specific task, we have implemented a “Triplet Loss Network”, due to its ability to learn similarities directly from the data and the embedding-based approach, which allows for efficient and scalable search (Schroff, Kalenichenko, and Philbin, 2015).
A Triplet Loss Network comprises three sub-networks with identical architecture and weights (Hoffer and Ailon, 2018).  Each sub-network takes a single input and produces a feature representation as its output, which is then used to learn the similarity and dissimilarity between inputs (Figure 2).
![alt text](https://github.com/Munkh99/iml/blob/master/figures/Screenshot%202023-06-05%20at%2010.51.22.png)
Figure 2: The Triplet Loss Network Architecture


For training, backpropagation is used through the Triplet Loss function:
![alt text](https://github.com/Munkh99/iml/blob/add7ba5bf8c96e691ad33bd617d1cb798cd29c99/figures/Screenshot%202023-06-05%20at%2011.03.27.png)



#### Transfer Learning
In the “Triplet Loss Network” architecture, each subnetwork has been constructed using Transfer Learning techniques. Specifically, for the task of image retrieval and comparison, three "twin" networks have been employed, all of which are pre-trained ResNet50 models. These pre-trained models serve as the basis for extracting and defining the features required to compare query images and have been chosen due to their great discriminative feature learning techniques (Figure 3).
![alt text](https://github.com/Munkh99/iml/blob/ba09d0c4eace11c4003c86d91664d37300ca80bf/figures/Screenshot%202023-06-05%20at%2010.51.37.png)
Figure 3: ResNet50 Architecture (He et al., 2015)

#### Inference Procedure
The inference process is performed by the `network.py` function, which takes the query set and the gallery set as inputs. It compares the extracted features from each image to determine the distance or dissimilarity between them. This distance estimation helps quantify the similarity or discrepancy between the query images and the gallery images. By analyzing the feature distances, the function can provide insights into the similarity rankings or identify potential matches between the query and gallery images (Figure 4).
![alt text](https://github.com/Munkh99/iml/blob/add7ba5bf8c96e691ad33bd617d1cb798cd29c99/figures/Screenshot%202023-06-05%20at%2010.51.46.png)
Figure 4: Inference procedure

### Contributions
The research into different pieces of literature and possible datasets was done as a group effort, to begin working on the project.  
The Triplet Loss Network has been mostly redefined in the architecture by Munkhdelger, while Pooria and Alice have dealt with data augmentation and face-cropping. 
The alternative approaches have been implemented by Pooria and Alice, however, the main code structure was changed and cleaned by Pooria.
The report was written by Alice, and figures and diagrams were drawn by Pooria and  Munkhdelger.

## References

- Halawani, A., Teynor, A., Brunner, G. and Burkhardt, H. (2006). Fundamentals and Applications of Image Retrieval: An Overview. Bioimaging and Analysis View project Ethics and Agile Software Development -Mapping the Landscape: A Mapping Review Protocol View project Fundamentals and Applications of Image Retrieval.
- Hameed, I.M., Abdulhussain, S.H. and Mahmmod, B.M. (2021). Content-based image retrieval: A review of recent trends. Cogent Engineering, 8(1), p.1927469. doi:https://doi.org/10.1080/23311916.2021.1927469.
- He, K., Zhang, X., Ren, S. and Sun, J. (2015). Deep Residual Learning for Image Recognition. [online] Available at: https://arxiv.org/pdf/1512.03385v1.pdf.
- Hoffer, E. and Ailon, N. (2018). Deep metric learning using Triplet network. arXiv:1412.6622 [cs, stat]. [online] Available at: https://arxiv.org/abs/1412.6622.
- Hosna, A., Merry, E., Gyalmo, J., Alom, Z., Aung, Z. and Azim, M.A. (2022). Transfer learning: a friendly introduction. Journal of Big Data, 9(1). doi:https://doi.org/10.1186/s40537-022-00652-w.
- Liu, Y., Zhang, D., Lu, G. and Ma, W.-Y. (2007). A survey of content-based image retrieval with high-level semantics. Pattern Recognition, 40(1), pp.262–282. doi:https://doi.org/10.1016/j.patcog.2006.04.045.
- Lu, D. and Weng, Q. (2007). A survey of image classification methods and techniques for improving classification performance. International Journal of Remote Sensing, [online] 28(5), pp.823–870. doi:https://doi.org/10.1080/01431160600746456.
- Schroff, F., Kalenichenko, D. and Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [online] doi:https://doi.org/10.1109/cvpr.2015.7298682.
- Suárez, J.L., García, S. and Herrera, F. (2021). A tutorial on distance metric learning: Mathematical foundations, algorithms, experimental analysis, prospects and challenges. Neurocomputing, 425, pp.300–322. doi:https://doi.org/10.1016/j.neucom.2020.08.017.
- Tan, M. and Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [online] arXiv.org. Available at: https://arxiv.org/abs/1905.11946.
- Wan, J., Hoi, S., Wu, P., Zhu, J., Wang, D., Wu, Zhang, Y. and Li (2014). Deep learning for content-based image retrieval: A comprehensive study. pp.157–166.
- Wang, J., Song, Y., Leung, T., Rosenberg, C., Wang, J., Philbin, J., Chen, B. and Wu, Y. (2014). Learning Fine-grained Image Similarity with Deep Ranking. [online] Available at: https://arxiv.org/pdf/1404.4661.pdf [Accessed 2 Jun. 2023].
- www.kaggle.com. (n.d.). CelebFaces Attributes (CelebA) Dataset. [online] Available at: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.






