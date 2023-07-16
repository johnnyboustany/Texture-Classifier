# Texture Classifier

Alongside two students, I built two base deep learning models for my Spring 2022 Deep Learning course to classify images based on texture and then fine-tuned the models to investigate how modifications to the architecture, tunable parameters, and amount of data passed in affected the learning and texture classification capabilities of the models.

<p align="center">
    <img src="./assets/texture.png" alt="" width="1000">
</p>

## Table of Contents
* [Technologies Used](#technologies-used)
* [General Info](#general-info)
* [Features](#features)
* [Results](#results)
* [Testing](#testing)
* [Usage Instructions](#usage-instructions)
* [Project Status](#project-status)
* [Conclusion](#conclusion)
* [Contributions](#contributions)
<!-- * [License](#license) -->

## Technologies Used
Python, TensorFlow
 
## General Info

We used the standard dataset DTD (Describable Textures Dataset), consisting of 56400 images, organized according to a list of 47 terms (categories) inspired from human perception. We reshaped and normalized the images, then split them into three groups: training, testing, and validation. We built two base models using ResNet and VGG and then modified them to optimize texture classification. In the end, we compared the performance of 3 ResNet models and 3 VGG models.

## Features

### ResNet Models:

The first base model was a ResNet that extracts 1000 features from the images, then passes the output into a fully connected dense layer with some pooling and dropouts. The second model had the same architecture as the first, but we decreased the batch size, and imputed four times as much data for the training, testing, and validation. The third ResNet model was identical to the first, but included two convolution layers.

### VGG Models:

We used VGG16 as a base model for all 3 of our VGG models. For our first VGG model, we only added 2D global average pooling and a single softmax-activated dense linear layer. For our second VGG model, we added two relu-activated 2D convolution layers, followed by 2D global average pooling. Then, after a flatten layer, we passed the outputs through 3 dense blocks. The dense block architecture consisted of a relu-activated dense layer followed by batch normalization and a dropout layer. The output was then passed through a final dense layer, activated with softmax. For our third VGG model, we used a convolution block consisting of two separable convolution layers with 32 output filters, followed by batch normalization and 2D Global Average Pooling. The output was then passed through 3 dense blocks and a final dense layer.

<p align="center">
    <img src="./assets/tabletexture.png" alt="">
</p>

<p align="center">
    <img src="./assets/graphtexture.png" alt="">
</p>

## Results

In order to evaluate the effectiveness of the models, we examined both precision and recall and used them to calculate F1-score (the harmonic mean of the two values). The first ResNet model performed the best and achieved the highest F1-score out of all the models. The second VGG model performed the worst out of all the models. This is consistent with the low accuracy and high loss observed for this model during the training and testing process. The first VGG model was the most successful VGG model, which is also consistent with its relatively high accuracy and low loss.

## Usage Instructions

### To run the Typeically program:
    
1. Preferably use Chrome for your browser and download the [CORS UnBlock extension](https://docs.google.com/document/d/1kAGzs_0YeLkAXbZUFNlNNj2SrcmW8tcc3CuH0Uy6cQ8/edit#heading=h.iiwoysfq2rkn). Activate it by making sure the yellow light on the bug icon is visible.

2. In one terminal:
```
cd frontend    
cd type-client
npm start
```
3. In another terminal:
```    
cd backend
./run --gui
```
### To test the Typeically program:
    
Run the Junit and Selenium tests using:
```
cd backend
mvn test
```
*IMPORTANT:*
- Make sure you activate the CORS UnBlock extension in the browser when the selenium tabs open
- No need to download the extension for testing, it is preloaded.

## Project Status
Project is: Complete (as of May 2022)

## Conclusion

I learned a lot about the software engineering cycle in general and realized how much I enjoyed working with both frontend and backend concurrently.

## Contributions

 I specifically worked on:
- the backend and frontend (React) aspects of the leaderboard, including setting up APIs to ensure the leaderboard is updated
- testing the leaderboard database from the backend by writing the JUnit tests
- the API handler to send the list of newly-released songs from Spotify to the frontend
- the frontend “choose a newly-released song” feature which allows users to choose a recent song
- testing all aspects of the frontend with Selenium
- helping my group members integrate their parts and making sure merges go smoothly

My group members: Dhiraj Khanal, Rakan Mosa O Alomran



# Textures-Classifier

## Authors: Johnny Boustany, Dhiraj Khanal, Rakan Mosa O Alomran

The goal of our project is to determine how different architectures affect the classification of texture. We built two base models using ResNet and VGG, and then fine-tuned them to investigate how modifications to the architecture, tunable parameters, and amount of data passed in affect the learning and texture classification capabilities of the models. 

The first base model was a ResNet that extracts 1000 features from the images, then passes the output into a fully connected dense layer with some pooling and dropouts. The second model had the same architecture as the first, but we decreased the batch size, and imputed four times as much data for the training, testing, and validation. The third ResNet model was identical to the first, but included two convolution layers.

We used VGG16 as a base model for all 3 of our VGG models. For our first VGG model, we only added 2D global average pooling and a single softmax-activated dense linear layer. For our second VGG model, we added two relu-activated 2D convolution layers, followed by 2D global average pooling. Then, after a flatten layer, we passed the outputs through 3 dense blocks. The dense block architecture consisted of a relu-activated dense layer followed by batch normalization and a dropout layer. The output was then passed through a final dense layer, activated with softmax. For our third VGG model, we used a convolution block consisting of two separable convolution layers with 32 output filters, followed by batch normalization and 2D Global Average Pooling. The output was then passed through 3 dense blocks and a final dense layer.


Here is the link to the [Project Writeup](https://docs.google.com/document/d/1nTk9OHvCTI8rGrsVXXEZt9qAQBM3ojPsbBitPrDigeY/edit?usp=sharing).

![A poster summarizing the project.](/docs/poster.png)
