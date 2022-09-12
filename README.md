# Textures-Classifier

## Authors: Johnny Boustany, Dhiraj Khanal, Rakan Mosa O Alomran

The goal of our project is to determine how different architectures affect the classification of texture. We built two base models using ResNet and VGG, and then fine-tuned them to investigate how modifications to the architecture, tunable parameters, and amount of data passed in affect the learning and texture classification capabilities of the models. 

The first base model was a ResNet that extracts 1000 features from the images, then passes the output into a fully connected dense layer with some pooling and dropouts. The second model had the same architecture as the first, but we decreased the batch size, and imputed four times as much data for the training, testing, and validation. The third ResNet model was identical to the first, but included two convolution layers.

We used VGG16 as a base model for all 3 of our VGG models. For our first VGG model, we only added 2D global average pooling and a single softmax-activated dense linear layer. For our second VGG model, we added two relu-activated 2D convolution layers, followed by 2D global average pooling. Then, after a flatten layer, we passed the outputs through 3 dense blocks. The dense block architecture consisted of a relu-activated dense layer followed by batch normalization and a dropout layer. The output was then passed through a final dense layer, activated with softmax. For our third VGG model, we used a convolution block consisting of two separable convolution layers with 32 output filters, followed by batch normalization and 2D Global Average Pooling. The output was then passed through 3 dense blocks and a final dense layer.


Here is the link to the [Project Writeup](https://docs.google.com/document/d/1nTk9OHvCTI8rGrsVXXEZt9qAQBM3ojPsbBitPrDigeY/edit?usp=sharing).

(/docs/example.png)
