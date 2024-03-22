# Stance Detection

Stance detection for tweets is an essential task in natural language processing (NLP), particularly in understanding public opinions and sentiments towards specific topics. This project focuses on developing a comprehensive stance detection system for tweets using machine learning algorithms and transformer-based models.

## Overview

The project follows the fundamental phases of any machine learning project, emphasizing the Training and Testing phases. It employs a structured architecture to handle complexities effectively. The main components of the architecture include data processing, generating embeddings using transformer models, and training/testing classifiers.

### Architecture
![Training Architecture](https://github.com/harshit-asu/Stance-Detection/blob/main/Visualization/Training.png)
![Testing Architecture](https://github.com/harshit-asu/Stance-Detection/blob/main/Visualization/Testing.png)

## Models

### BERT
[BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer model in NLP, renowned for its performance on various benchmarks. It excels in extracting information and learning sequential features within text. We utilize the BERT model for generating text embeddings.

### ELECTRA
[ELECTRA](https://arxiv.org/abs/2003.10555) is a BERT variant that addresses some limitations of its predecessor. It follows a generator-discriminator style pre-training, which improves learning efficiency and co-relation understanding between tokens compared to BERT. We utilize the ELECTRA model for generating text embeddings.

## Classifiers

The project employs Support Vector Machines (SVM) and Long Short-Term Memory (LSTM) networks as classifiers. SVM is chosen for its superior performance in various tasks, while LSTM excels in capturing longer sequential representations.

## Evaluations

The performance of the models is evaluated based on key metrics such as Precision, F1-score, Recall, and Accuracy. The evaluation provides insights into the efficacy of stance detection across different target categories and models.

### Results
The results section presents a comprehensive overview of stance detection performance across various target categories and models. It highlights accuracy percentages for each combination, offering valuable insights into the strengths and weaknesses of the implemented models.

![Results](https://github.com/harshit-asu/Stance-Detection/blob/main/Visualization/result.png)

## User Interface

The Stance Detection web application provides an intuitive interface for users to input tweets and select target categories and models for detection. The results are dynamically generated and presented in a visually appealing format, enhancing the overall user experience.

## Examples

![Example 1: Against](https://github.com/harshit-asu/Stance-Detection/blob/main/Visualization/Against_UI.png)
![Example 2: Favor](https://github.com/harshit-asu/Stance-Detection/blob/main/Visualization/Favor_UI.png)
![Example 3: Neutral](https://github.com/harshit-asu/Stance-Detection/blob/main/Visualization/Neutral_UI.png)

## Team Members: 

Harshit Allumolu (hallumol@asu.edu)

Pravalika Gollapudi (pgollap2@asu.edu)

Sri Naga Sushma Jyothula (sjyothul@asu.edu)

Sumeet Choudhary (schoud36@asu.edu)

Venkata Naga Aditya Datta Chivukula (vchivuk4@asu.edu)

Venkata Pavan Boppudi (vboppud1@asu.edu)
