# Hybrid Language Model
This repository contains code for building a hybrid language model from scratch using TensorFlow. The hybrid model combines a recurrent neural network (RNN) with a transformer model to model the dependencies between words and sentences in a text dataset.

## Requirements
•Python 3.6 or higher
•TensorFlow 2.x
•requests
•bs4
## Usage

To build and train the hybrid language model, follow these steps:

1.Collect and preprocess the text data by running the `collect_text_data.py` script:
```c
python collect_text_data.py
```
This script collects text data from a website, extracts the text from the HTML page, cleans the text by removing unwanted characters, and tokenizes the text into words or subwords. It also performs other preprocessing steps such as lowercasing, stemming, or lemmatization.

2.Build the hybrid model by running the `build_model.py` script:
```c
python build_model.py
```
This script defines the hybrid model architecture, which consists of an embedding layer, an LSTM RNN layer, a transformer layer, and a dense output layer. It also defines the input and output shapes of the model.

3.Train the model by running the `train_model.py` script:
```c
python train_model.py
```
This script compiles the model using the Adam optimizer and the sparse categorical cross-entropy loss function, and then fits the model to the training data using mini-batch stochastic gradient descent.

4.Evaluate the model by running the `evaluate_model.py` script:
```c
python evaluate_model.py
```
This script evaluates the model on the test data and prints the loss and accuracy metrics.
