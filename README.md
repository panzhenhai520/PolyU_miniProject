Paper: "ADBench: Anomaly Detection Benchmark" https://arxiv.org/abs/2206.09426

Code: https://github.com/Minqi824/ADBench?tab=readme-ov-file#readme

## I. Paper datasets: 25_musk, 3_backdoor, 9_census

- 25_musk: 166 features, 3062 samples
- 3_backdoor: 196 features, 95329 samples
- 9_census: 500 features, 299285 samples

## II. Ten algorithms for anomaly detection under fully supervised mode in the paper:  
Logistic Regression, Naive Bayes, Support Vector Machine, Multi-layer Perceptron, Random Forest, LightGBM, XGBoost, CatBoost, ResNet, FTTransformer

## III. Runtime environment of the code:   
Python 3.7.16, Pytorch 1.13, Pytorch-cuda 11.7, Tensorflow 2.11, Keras 2.11.0, Numpy 1.21.5, Pandas 1.3.5, Matplotlib 3.53, Tqdm 4.66.2

## IV. Modifications to the ADBench (Anomaly Detection Benchmark) paper code's：  
data_generator to exclude Image and NLP type data, and to change the comparison from 57 datasets to 3 datasets, modify 6 types of noise to one, with a noise coefficient of 0.5. After the aforementioned modifications, the total time spent on machine learning completion: 108048.69s

## V. Entirely rewritten minProject project code   
completing machine learning for five algorithms: Random Forest, SVM, Pytorch LSTM, Pytorch GRU, tensorflow LSTM, on the three datasets from the paper, and generating loss function curve graphs during training, calculating, and model performance curves.
