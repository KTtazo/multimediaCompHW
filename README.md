# multimediaCompHW
Repository for multimedia computing homework spring 2022

With resnet we aggregate new layers to the neural networkis such a way we put te so called residual connections among the layers.
This connections is not folllowing the straight order in a succesive way, but it connects not-traditionally conected layers by jumping one or more levels. 
By using this technique, we can ensure a much better performance than the average neural networks with traditional connections.
The result of the jumping of two layers is added to the function "relu", the activation function.
In  neural network, when adding layers without residual connections, there is a point where the error instead of correcting and reducing, it starts to increase.
The gradient of the addition with the activation function will not only propagate backwards to the previous layer, but also in the direction of the residual conection


-------First part of the homework-----------
We first download the database of pictures that we will use in the feature extraction. 
By using the following link (https://www.tensorflow.org/datasets/catalog/caltech101), or many others that can be found in the net, we obtain a big list of files containing an average of two to three hundred .jpg image files that will be used in the program. 
We create a model using ResNet50 residual network and normalize the features obtained one by one in evey image present in the database. The features are then stored in a picke file called "features-caltech101.pkl". This process of extracting the features and storing them is going to be made only once in order to avoid generating the same file with the same values over again since the process is very time consuming.  
