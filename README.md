# multimediaCompHW
Repository for multimedia computing homework spring 2022

With resnet we aggregate new layers to the neural networkis such a way we put te so called residual connections among the layers.
This connections is not folllowing the straight order in a succesive way, but it connects not-traditionally conected layers by jumping one or more levels. 
By using this technique, we can ensure a much better performance than the average neural networks with traditional connections.
The result of the jumping of two layers is added to the function "relu", the activation function.
In  neural network, when adding layers without residual connections, there is a point where the error instead of correcting and reducing, it starts to increase.
The gradient of the addition with the activation function will not only propagate backwards to the previous layer, but also in the direction of the residual conection
