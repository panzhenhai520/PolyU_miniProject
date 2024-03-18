import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from itertools import product
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import pandas as pd
from keras.callbacks import Callback
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datasets.data_generator import DataGenerator  # 来自ADBench的数据预处理包 做了适当修改
import argparse
import sys


class LSTMBinaryClassifier(nn.Module):
    #  LSTM model in Pytorch
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, 50, batch_first=True)
        self.dense1 = nn.Linear(50, 10)
        self.dense2 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # 注意！该方法是隐式调用的，在nn.Module 的__call__ 中实现
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # 如果x含时间维度,需要 x = self.dense1(x[:, -1, :])
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

    def predict_score(self,x):
        #接收张量x 作为输入,返回正类的概率分数
        self.eval()
        with torch.no_grad():
            score=self(x)
            score=score.squeeze()
        return score

class GRUBinaryClassifier(nn.Module):
    # GRU model in Pytorch
    def __init__(self, input_size, hidden_size):
        super(GRUBinaryClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output)  # 直接传给全连接层
        output = self.sigmoid(output)
        return output

def dataset_filter():
    # dataset list in the current folder
    dataset_list_org = data_generator.generate_dataset_list()
    data_generator.dataset_list_classical =dataset_list_org
    dataset_list, dataset_size = [], []
    generate_duplicates =True
    n_samples_threshold = 1000
    mode='rla'
    nla_list = [0, 1, 5, 10, 25, 50, 75, 100]
    realistic_synthetic_mode=None
    noise_type=None
    seed_list = [1]
    for dataset in dataset_list_org:
        for seed in seed_list:
            add = True
            data_generator.seed = 1
            data_generator.dataset = dataset
            data = data_generator.generator(la=1.00, at_least_one_labeled=True)
            if data['y_train'].size == 0 and data['y_test'].size == 0:
                add = False
            if not generate_duplicates and len(data['y_train']) + len(data['y_test']) < n_samples_threshold:
                add = False
            else:
                if mode == 'nla' and sum(data['y_train']) >= nla_list[-1]:
                    pass
                elif mode == 'rla' and sum(data['y_train']) > 0:
                    pass
                else:
                    add = False
            # remove high-dimensional CV and NLP datasets if generating synthetic anomalies or robustness test
            if realistic_synthetic_mode is not None or noise_type is not None:
                if isin_NLPCV(dataset):
                    add = False
            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
    # sort datasets by their sample size
    dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]
    return dataset_list

def isin_NLPCV(dataset):
    if dataset is None:
        return False
    else:
        NLPCV_list = ['agnews', 'amazon', 'imdb', 'yelp', '20news',
                        'MNIST-C', 'FashionMNIST', 'CIFAR10', 'SVHN', 'MVTec-AD']
    return any([_ in dataset for _ in NLPCV_list])

def show_history(history):
    # 画出训练损失('loss')和测试损失('val_loss')
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']
    epochs = range(1, len(loss) + 1)

    # 确保 val_loss 和 val_acc 的长度与 epochs 相匹配
    if len(val_loss) != len(epochs) or len(val_acc) != len(epochs):
        print("验证损失或准确率的数据长度与训练周期不匹配。")
        return

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 提取训练准确率 ('acc') 和测试准确率 ('val_acc')
    acc = history['acc']
    val_acc = history['val_acc']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Test acc')
    plt.title('Training and Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def roc_f(y_data, y_score, title):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_data, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' RNN-LSTM Model ')
    plt.legend(loc="lower right")
    plt.show()

def plot_multiple_roc_curves(data_pairs, titles, overall_title):
   # draw multiple ROC curves
    plt.figure()
    lw = 2

    for (y_data, y_score), title in zip(data_pairs, titles):
        # 计算每一类的ROC曲线和ROC面积
        fpr, tpr, threshold = roc_curve(y_data, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f'{title} ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(overall_title)
    plt.legend(loc="lower right")
    plt.show()

class TimeHistory(Callback):
    # 在keras_lstm_model模型中每个epoch训练时间的回调函数
    def on_train_begin(self, logs={}):
        self.train_times = []
        self.val_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.train_times.append(time.time() - self.epoch_start_time)
        if 'val_loss' in logs:
            self.val_times.append(time.time() - self.epoch_start_time - self.train_times[-1])

class PytrochData(object):
    # 把数据处理成 Pytroch 需要的张量形式
    def __init__(self, X_train,y_train,X_test,y_test):
        self.device = data_generator.get_device(gpu_specific=True,Showinfo=False)
        self.X_train=X_train
        self.y_train =y_train
        self.X_test =X_test
        self.y_test=y_test
        self.X_train_tensor = torch.tensor(X_train).float().to(self.device)
        self.y_train_tensor = torch.tensor(y_train).unsqueeze(1).float().to(self.device)
        self.X_test_tensor = torch.tensor(X_test).float().to(self.device)
        self.y_test_tensor = torch.tensor(y_test).unsqueeze(1).float().to(self.device)
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

class ModelFactory:
     # 创建模型(工厂模式)
     def __init__(self, model_name, algorithm_type,epochs,PData):
        self.device=PData.device
        self.model_name=model_name
        self.algorithm_type=algorithm_type
        self.epochs=epochs
        self.PData =PData
        self.y_test_pred=None
        self.y_test_pred_proba=None
        self.time_fit=None
        self.time_inference=None
        self.model=None
        self.y_train_pred_proba=None
        self.overall_title=None
        self.metrics=[]

     def get_model(self):
         if self.model_name == 'Pytorch_model':
            # 添加一个时间步长维度，使其成为符合LSTM三维张量的形式： [batch_size, 1, features]
            X_train_tensor = self.PData.X_train_tensor.unsqueeze(1)
            if algorithm_type == 'LSTM':
               self.model=LSTMBinaryClassifier(X_train_tensor.shape[2], 32)
            elif algorithm_type == 'GRU':
               self.model=GRUBinaryClassifier(X_train_tensor.shape[2], 32)
            self.device = data_generator.get_device(gpu_specific=True,Showinfo=False)

         elif model_name == 'keras_lstm_model':
             # 采用tensorflow 编写的LSTM 模型
             import tensorflow as tf
             if tf.config.experimental.list_physical_devices('GPU'):
                 print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
             else:
                 print("GPU is not detected")
             gpus = tf.config.list_physical_devices('GPU')
             if gpus:
                 for i, gpu in enumerate(gpus):
                     print(f'GPU {i}: Name: {gpu.name}, Type: {gpu.device_type}')
             self.device= '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

             time_callback = TimeHistory()
             with tf.device(self.device):
                 # 定义模型
                 lstm = Sequential()
                 lstm.add(LSTM(units=32, return_sequences=True, input_shape=(self.PData.X_train.shape[1], 1)))
                 lstm.add(LSTM(50))
                 lstm.add(Dense(10, activation='relu'))
                 lstm.add(Dense(1, activation='sigmoid'))
                 lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
                 self.model=lstm

         elif model_name == 'SVM_model':
             # 支持向量机
             from sklearn.svm import SVC
             self.model= SVC(kernel='linear', probability=True)

         elif model_name == 'RandomForest_model':
             # 随机森林模型
             from sklearn.ensemble import RandomForestClassifier
             self.model=RandomForestClassifier(n_estimators=100, max_depth=10)

         else:
             raise ValueError("模型未定义")

         return self.model

     def model_flt(self):
         from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
         results = []
         history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'train_time': [], 'val_time': []}

         if model_name =='Pytorch_model':
             # 加载训练集数据
             train_loader = DataLoader(dataset=self.PData.train_dataset, batch_size=64, shuffle=True)
             # 加载测试集数据
             test_loader = DataLoader(dataset=self.PData.test_dataset, batch_size=64, shuffle=False)
             model.to(self.device)
             optimizer = Adam(model.parameters(), lr=0.001)
             criterion = nn.BCELoss()

             for epoch in range(self.epochs):
                 start_time_epoch = time.time()
                 model.train()
                 train_losses, train_accuracies = [], []
                 for inputs, labels in train_loader:
                     inputs, labels = inputs.to(self.device), labels.to(self.device)
                     optimizer.zero_grad()
                     outputs = model(inputs)
                     loss = criterion(outputs, labels)
                     loss.backward()
                     optimizer.step()

                     train_losses.append(loss.item())
                     predictions = torch.round(outputs)
                     accuracy = torch.sum(predictions == labels).item() / labels.size(0)
                     train_accuracies.append(accuracy)
                 end_time_train = time.time()
                 time_fit = end_time_train - start_time_epoch

                 model.eval()
                 val_losses, val_accuracies = [], []
                 start_time_val = time.time()
                 with torch.no_grad():
                     for inputs, labels in test_loader:
                         inputs, labels = inputs.to(self.device), labels.to(self.device)
                         outputs = model(inputs)
                         loss = criterion(outputs, labels)
                         val_losses.append(loss.item())
                         predictions = torch.round(outputs)
                         accuracy = (predictions == labels).float().mean().item()
                         val_accuracies.append(accuracy)
                 end_time_val = time.time()
                 time_inference = end_time_val - start_time_val

                 val_loss_avg = sum(val_losses) / len(val_losses)
                 val_accuracy_avg = sum(val_accuracies) / len(val_accuracies)

                 history['loss'].append(np.mean(train_losses))
                 history['acc'].append(np.mean(train_accuracies))
                 history['val_loss'].append(np.mean(val_losses))
                 history['val_acc'].append(np.mean(val_accuracies))
                 history['train_time'].append(time_fit)
                 history['val_time'].append(time_inference)

                 print(f'Epoch {epoch + 1}, Loss: {np.mean(train_losses):.4f}, '
                       f'Acc: {np.mean(train_accuracies):.4f}, '
                       f'Val Loss: {np.mean(val_losses):.4f}, '
                       f'Val Acc: {np.mean(val_accuracies):.4f}, '
                       f'TrainTime: {time_fit:.4f}s, '
                       f'Val Time: {time_inference:.4f}s')

             total_train_time = sum(history['train_time'])
             total_val_time = sum(history['val_time'])
             self.time_fit = total_train_time
             self.time_inference = total_val_time

             print(f'Total training time: {total_train_time:.4f}s, Total validation time: {total_val_time:.4f}s')

             # 禁用梯度计算
             with torch.no_grad():
                 logits_train = model(self.PData.X_train_tensor).squeeze()
                 y_train_score=torch.sigmoid(logits_train)
                 logits_test = model(self.PData.X_test_tensor).squeeze()
                 y_test_score=torch.sigmoid(logits_test)
                 self.y_train_pred_proba = torch.sigmoid(logits_train).cpu().numpy()
                 self.y_test_pred_proba = torch.sigmoid(logits_test).cpu().numpy()
                 predictions = model(self.PData.X_test_tensor)
                 y_pred = torch.round(predictions)

                 # 将存储在GPU上的张量转换为NumPy数组之前，先将张量移动回CPU内存
                 y_pred_np = y_pred.cpu().numpy().flatten()
                 self.y_test_pred=y_pred_np

                 self.y_test_pred = y_pred.cpu().numpy().flatten()  # 保存测试集的预测分类结果
                 y_train_score_np = y_train_score.cpu().numpy()  # 保存训练集的预测概率
                 y_test_score_np = y_test_score.cpu().numpy()


             y_test_np = self.PData.y_test_tensor.cpu().numpy().flatten()
             precision = precision_score(y_test_np, self.y_test_pred)
             recall = recall_score(y_test_np, self.y_test_pred)
             f1 = f1_score(y_test_np, self.y_test_pred)

             print("precision: {:.2f}%".format(precision * 100))
             print("recall_score: {:.2f}%".format(recall * 100))
             print("F1 Score: {:.2f}%".format(f1 * 100))
             y_train_score_np = y_train_score.cpu().numpy()
             y_test_score_np = y_test_score.cpu().numpy()
             score_test = y_test_score

             # 确保y_true、y_score是NumPy数组，并且已经从CUDA移动到CPU
             #y_true_np = data['y_test'].cpu().numpy() if isinstance(data['y_test'], torch.Tensor) else data['y_test']
             y_true_np = self.PData.y_test_tensor.cpu().numpy() if isinstance(self.PData.y_test_tensor,
                                                                              torch.Tensor) else self.PData.y_test
             y_score_np = score_test.cpu().numpy() if isinstance(score_test, torch.Tensor) else score_test

             # 调用metric函数，传入NumPy数组
             self.metrics = data_generator.metric(y_true=y_true_np, y_score=y_score_np, pos_label=1)
             results.append([params, 'Pytorch_' + self.algorithm_type, self.metrics, total_train_time, total_val_time])
             show_history(history)
             print(f"\nModel: Pythorch_{self.algorithm_type}, AUC-ROC: {self.metrics['aucroc']}, AUC-PR: {self.metrics['aucpr']}")
             # return y_train_score_np, y_test_score_np, self.metrics, results, history, total_train_time, total_val_time

         elif model_name =='keras_lstm_model':
             from keras.utils import plot_model
             time_callback = TimeHistory()
             hist = model.fit(self.PData.X_train, self.PData.y_train, validation_data=(self.PData.X_test, self.PData.y_test),
                             epochs=self.epochs, batch_size=64, callbacks=[time_callback])
             history = {
                 'loss': hist.history['loss'],
                 'val_loss': hist.history['val_loss'],
                 'acc': hist.history['acc'],
                 'val_acc': hist.history['val_acc'],
                 'train_time': time_callback.train_times,
                 'val_time': time_callback.val_times,
             }
             score = model.evaluate(self.PData.X_test, self.PData.y_test, batch_size=128)
             total_train_time = sum(time_callback.train_times)
             total_val_time = sum(time_callback.val_times)

             y_train_score = model.predict(self.PData.X_train).ravel()
             self.y_train_pred_proba = y_train_score
             y_test_score = model.predict(self.PData.X_test).ravel()
             self.y_test_pred_proba = y_test_score
             self.y_test_pred = (y_test_score > 0.5).astype(int)

             self.metrics = data_generator.metric(self.PData.y_test, y_test_score, pos_label=1)
             results.append([params, 'keras_lstm', self.metrics, total_train_time, total_val_time])
             print('----- 生成模型结构图model.png -----')
             model.summary()
             plot_model(model, to_file='model.png')
             show_history(history)
             print(f"Model: keras_LSTM, AUC-ROC: {self.metrics['aucroc']}, AUC-PR: {self.metrics['aucpr']}")

             self.time_fit=total_train_time
             self.time_inference=total_val_time
             print(self.time_fit)
             print(self.time_inference)
             # return y_train_score, y_test_score, metrics, results, history, total_train_time, total_val_time

         elif model_name == 'RandomForest_model' or 'SVM_model':
             start_time = time.time()
             self.model.fit(self.PData.X_train, self.PData.y_train)
             end_time = time.time()
             self.time_fit = end_time - start_time
             start_time = time.time()
             self.y_test_pred = self.model.predict(self.PData.X_test)
             self.y_test_pred_proba = self.model.predict_proba(self.PData.X_test)[:, 1]
             end_time = time.time()
             self.time_inference = end_time - start_time

     def model_performance(self):
         from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
         from sklearn.metrics import average_precision_score

         test_accuracy = accuracy_score(self.PData.y_test, self.y_test_pred)
         test_precision = precision_score(self.PData.y_test, self.y_test_pred,zero_division=1)

         test_recall = recall_score(self.PData.y_test, self.y_test_pred)
         test_f1 = f1_score(self.PData.y_test, self.y_test_pred)

         if model_name == 'RandomForest_model' or model_name =='SVM_model':
             self.y_train_pred_proba = model.predict_proba(self.PData.X_train)[:, 1]
             self.y_test_pred_proba = model.predict_proba(self.PData.X_test)[:, 1]
             train_fpr, train_tpr, _ = roc_curve(self.PData.y_train, self.y_train_pred_proba)
             train_aucroc= auc(train_fpr, train_tpr)
             train_aucpr = average_precision_score(self.PData.y_train, self.y_train_pred_proba)
         elif model_name =='Pytorch_model' or model_name =='keras_lstm_model':
             train_fpr, train_tpr, _ = roc_curve(self.PData.y_train.ravel(), self.y_train_pred_proba)
             train_aucroc= auc(train_fpr, train_tpr)
             train_aucpr = average_precision_score(self.PData.y_train.ravel(), self.y_train_pred_proba)

         print(
             f"Test Accuracy: {test_accuracy:.4%}, Precision: {test_precision:.4%}, Recall: {test_recall:.4%}, F1 Score: {test_f1:.4%}")
         if model_name=='Pytorch_model':
             self.overall_title='Pytorch_'+algorithm_type
         else:
             self.overall_title =model_name

         self.metrics = {'aucroc': train_aucroc, 'aucpr': train_aucpr}

         plot_multiple_roc_curves(data_pairs=[(self.PData.y_train.ravel(), self.y_train_pred_proba.ravel()), (self.PData.y_test.ravel(), self.y_test_pred_proba.ravel())],
                                  titles=['Training', 'Test'],
                                  overall_title=self.overall_title)
         return self.metrics

def main(model_name, algorithm_type, epochs):
    print(f"模型名称: {model_name}")
    print(f"算法类型: {algorithm_type}")
    print(f"Epochs: {epochs}")

    # model_List=['Pytorch_model','keras_lstm_model','RandomForest_model','SVM_model']
    # model_name = model_List[1]
    # algorithm_type='LSTM'
    # epochs=50
    rla_list = [1.00]
    seed_list = [1]
    noise_params_list = [0.50]
    noise_type = 'irrelevant_features'
    realistic_synthetic_mode = None
    noise_param = [0.1]

    # 实例化data_generator包中的数据处理类完成数据预处理和标准化
    data_generator = DataGenerator(generate_duplicates=True,n_samples_threshold=1000)
    experiment_params=[]

    dataset=None  # None表示不使用自己的数据集，用datasets 目录下的npz实验数据
    if dataset is None:
        dataset_list = dataset_filter()
        X, y = None, None
    else:
        isinstance(dataset, dict)
        dataset_list = [None]
        X = dataset['X']; y = dataset['y']

    if noise_type is not None:
        experiment_params = list(product(dataset_list, rla_list, noise_params_list, seed_list))
    else:
        experiment_params = list(product(dataset_list, rla_list, seed_list))

    print(f'{len(dataset_list)} datasets')
    print(f"\nExperiment results are saved at: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')}")
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'), exist_ok=True)

    if model_name=='Pytorch_model':
      column_name = model_name + '_' + algorithm_type
    else:
      column_name = model_name

    df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=[column_name])
    df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=[column_name])
    df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=[column_name])
    df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=[column_name])

    for i, params in tqdm(enumerate(experiment_params)):
        print('--------------------------------------------------------------------------')
        print(f"This is No{i+1} data, params: {params} ,using {column_name}")
        print('--------------------------------------------------------------------------')

        if noise_type is not None:
            dataset, la, noise_param, seed = params
        else:
            dataset, la, seed = params
        if isin_NLPCV(dataset) and seed > 1:
            continue
        data_generator.seed = seed
        data_generator.dataset = dataset
        # 加载到data
        data = data_generator.generator(la=1.0, at_least_one_labeled=True, X=X, y=y,
                                        realistic_synthetic_mode=realistic_synthetic_mode,
                                        noise_type=noise_type, noise_ratio=noise_param)
        X_train=data['X_train']
        y_train=data['y_train']
        X_test=data['X_test']
        y_test=data['y_test']

        ssData = PytrochData(X_train, y_train, X_test, y_test)
        MyModel = ModelFactory(model_name, algorithm_type,epochs,PData=ssData)
        model = MyModel.get_model()
        MyModel.model_flt()
        metrics=MyModel.model_performance()

        # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
        df_AUCROC[column_name].iloc[i] = metrics['aucroc']
        df_AUCPR[column_name].iloc[i] = metrics['aucpr']
        df_time_fit[column_name].iloc[i] = MyModel.time_fit
        df_time_inference[column_name].iloc[i] = MyModel.time_inference

    df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'result', 'AUCROC_Mymodel'  + '.csv'), index=True)
    df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'result', 'AUCPR_Mymodel' +  '.csv'), index=True)
    df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'result', 'Time(fit)_Mymodel'  + '.csv'), index=True)
    df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'result', 'Time(inference)_Mymodel' + '.csv'), index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型训练参数')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--algorithm', type=str, default='', help='算法类型，仅当选择Pytorch_model时需要')
    parser.add_argument('--epochs', type=int, required=True, help='训练周期数')
    args = parser.parse_args()
    main(args.model, args.algorithm, args.epochs)