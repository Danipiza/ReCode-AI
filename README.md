
*[ENGLISH](README.md) ∙ [ESPAÑOL](https://github.com/Danipiza/ReCode-AI/README_ESP.md)* <img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=danipiza.ReCode-AI" />

<hr>
<h1 align="center"> ReCode-AI</h1>

<h2 align="center">A repository dedicated to artificial intelligence techniques. </h3>
<hr>


# INDEX
0. [Repo Structure]()
1. [Machine Learning](#machine-learning)
2. [Deep Learning](#deep-learning)
3. [Genetic Algorithms](#genetic-algorithms)
4. [Natural Language Processing (NLP)](#natural-language-processing-nlp)
5. [Computer Vision](#computer-vision)


# Repo Structure
```ruby
/
├── 0_Libraries/
│   ├── Keras/
│   │   └── # TODO
│   └── Pytorch/ # TODO          
│       ├── tensors.ipynb
│       └── image_datasets.ipynb
├── 1_ML/
│   ├── Reinforcement_Learning/Games-AI
│   ├── Supervised_Learning/
│   │   ├── Linear_Regression/
│   │   ├── K_Nearest_Neigbors/
│   │   └── Neural_Network/
│   └── UnSupervised_Learning/
│       ├── Agglomerative_Hierarchical/
│       └── KMeans/
│
├── 2_DL/ # TODO
│   ├── Neural_Network/
│   │   └── Prácticas/
│
├── 3_Genetic/ # TODO
│   ├── Binary_Real_problems/
│   │   ├──
│
├── 4_NLP/# TODO
│   ├──
│
└── 5_Computer_Vision/ # TODO
    ├── Captcha_solver/
    └── Img_analyzer/
```

<hr>


# Machine Learning

1. [Reinforcemen Learning](#reinforcement-learning)
2. [Supervised Learning](#supervised-learning)
3. [UnSupervised Learning](#unsupervised-learning)

## Reinforcement Learning


An agent learns to make best decisions by interacting with the environment. The goal of the agent is to maximize the obtained rewards over the execution time.

- Agent: moves in the environment executing actions.
- Environment: everything the agent interacts with.
- State: information of the current situation of the execution.
- Actions: the agent execute a sort of actions to interact with the environment. 
- Rewards: positive or negatives values given to the agent for executing an action in the current state.
- Policy: the objetive of the agent is to learn a policy that maximize the total rewards over time.

<br>

<h2 align="center" href=""> 
  <a href="https://github.com/Danipiza/Games-AI">Games-AI</a>   
</h2>

<div align="center">
  <img src="https://raw.githubusercontent.com/Danipiza/danipiza.github.io/refs/heads/main/public/Games_AI.webp" alt="Games-AI image" width="500"/>
</div>

### Algorithms
- Deep Q-Networks (DQN)
- Soft Actor Critic (SAC)
- Proximal Policy Optimization (PPO)



<br>

## Supervised Learning
An algorithm is trained on labeled data. The goal is for the algorithm to learn a mapping between input features and output labels so that it can make accurate predictions on new unseen data. It is called "supervised" because the model learns under the supervision of a labeled dataset, meaning it has access to the correct outputs (labels) for the training examples.

### Algorithms
- [Linear Regression](#linear-regression)
- [K-Nearest Neigbors (KNN)](#k-nearest-neigbors-knn)
- [Neural Network](#neural-network)

### [Linear Regression](https://github.com/Danipiza/ReCode-AI/tree/main/1_ML/Supervised_Learning/Linear_Regression)
A method to help understand the relationship between two variables:
- The predictor (independent) variable x sometimes called a feature
- The target (dependent) variable y
<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/97f944f6-7106-4230-8639-ebf6c4a78a15" alt="Linear regression image" width="400"/>
</div>


<br>

### [K-Nearest Neigbors (KNN)](https://github.com/Danipiza/ReCode-AI/tree/main/1_ML/Supervised_Learning/K_Nearest_Neigbors)
Used for classification and regression tasks. Operates on the idea that similar data exist in close proximity. In other words, objects that are close together in a feature space are likely to belong to the same class or have similar values.

<br>

### [Neural Network](https://github.com/Danipiza/ReCode-AI/tree/main/1_ML/Supervised_Learning/Neural_Network.py)
A computational model inspired by the way biological neural networks in the brain process information. It is a key component of deep learning and is widely used in tasks like classification, regression, image recognition, natural language processing, and more.


<br>

## UnSupervised Learning
The algorithm is trained on data without any labels or output variables. The goal is to discover patterns, structures, or relationships within the data. 
### Algorithms
- [Agglomerative Hierarchical](#agglomerative-hierarchical)
- [K-Means](#k-means)



### [Agglomerative Hierarchical](https://github.com/Danipiza/ReCode-AI/tree/main/1_ML/UnSupervised_Learning/Agglomerative_Hierarchical.py)
Partitional clustering algorithm, which builds a tree-like structure of clusters. The agglomerative approach starts with each data point as its own cluster and then merges the closest clusters step-by-step until all points are in a single cluster or until a desired number of clusters is achieved.


### [K-Means](https://github.com/Danipiza/ReCode-AI/tree/main/1_ML/UnSupervised_Learning/KMeans.py)
Partitional clustering algorithm that divides the data into a pre-specified number of clusters _K_. It is a centroid-based algorithm, meaning each cluster is represented by the mean (centroid) of the points in the cluster.

<br>

# Deep Learning


<br>
<hr>


# Genetic Algorithms


<br>
<hr>

# Natural Language Processing (NLP)


<br>
<hr>


# Computer Vision

## CNN
### [Image Analyzer](https://github.com/Danipiza/ReCode-AI/tree/main/5_Computer_Vision/Img_analyzer)
This program utilizes Convolutional Neural Networks (CNNs) to analyze images and classify them into predefined categories. The categories of the images are types of clothes. This repository includes the model architecture, data preprocessing, training scripts, and evaluation methods.

<div align="center">
  <img src="https://raw.githubusercontent.com/Danipiza/ReCode-AI/refs/heads/main/5_Computer_Vision/Img_analyzer/data/img/fashion6.png" alt="Games-AI image" width="200"/>
</div>

- [Pytorch](https://github.com/Danipiza/ReCode-AI/blob/main/5_Computer_Vision/Img_analyzer/pytorch.py)

### [Captcha Solver](https://github.com/Danipiza/ReCode-AI/tree/main/5_Computer_Vision/Captcha_solver)
This program is designed to solve alphanumeric CAPTCHAs using Convolutional Neural Networks (CNNs). By training a model on a dataset of CAPTCHA images, the program can recognize and predict the characters displayed in the CAPTCHAs, enabling automated form submissions or access to restricted content.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5db5b918-8f2c-4f65-8189-74a80466096a" alt="Games-AI image" width="200"/>
</div>

- [Keras](https://github.com/Danipiza/ReCode-AI/blob/main/5_Computer_Vision/Captcha_solver/keras.py)
- [Pytorch](https://github.com/Danipiza/ReCode-AI/blob/main/5_Computer_Vision/Captcha_solver/pytorch.py)

<br>
<hr>