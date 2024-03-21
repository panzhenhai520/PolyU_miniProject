Paper: "ADBench: Anomaly Detection Benchmark" https://arxiv.org/abs/2206.09426
Code: https://github.com/Minqi824/ADBench?tab=readme-ov-file#readme

1.论文数据集： 25_musk 、3_backdoor、9_ census
	25_musk: 166 个特征，3062 个样本
	3_backdoor : 196 个特征，95329 个样本
	9_ census : 500 个特征 299285 个样本

2.论文中全监督模式下异常检测的十种算法：
Logistic Regression、Naive Bayes、Support Vector Machine、Multi-layer Perceptron、Random Forest、LightGBM、XGBoost、CatBoost、ResNet、FTTransformer

3.文代码的运行环境:
   Python3.7.16, Pytorch 1.13, Pytorch-cuda 11.7  Tensorflow2.11, Keras 2.11.0,Numpy 1.21.5 ,Pandas1.3.5 ,Matplotlib3.53 Tqdm4.66.2
   
4.修改 ADBench(Anomaly Detection Benchmark) 论文源代码，排除 Image 和NLP 类型数据，并将程序对 57 个数据集的比较改为只针对 3 个数据集进行比较，将 6 种噪声修改为一种，噪声系数 0.5, 经过上述修改后，机器学习完成总花费时长：108048.69s

5.全新编写 minProject 项目代码，完成了五种算法：Random Forest、SVM、Pytorch LSTM、Pytorch GRU、 tensorflow LSTM ,在论文的三种数据集上的机器学习，并生成训练过程中的损失函数曲线图，计算和模型性能曲线

