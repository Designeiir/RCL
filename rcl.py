import pickle
from config import Config
import os
import random
from data_preprocessing import data_preprocessing, anomaly_detect, remove_region_latency
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
import torch.optim as optim
import numpy as np
import time

class MyDataSet(Dataset):  # 定义类，用于构建数据集
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length


class TransformerEncoderClassification(nn.Module):
    def __init__(self, dimension, head_num, layer_num, sequence_length, out_feature):
        super(TransformerEncoderClassification, self).__init__()
        # 先用nn.TransformerEncoderLayer构造Encoder层，再用nn.TransformerEncoder构造最终的Encoder，最终的Encoder可以包括几个层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dimension, nhead=head_num),  # 两个参数分别是输入样本的维度(50)和Encoder头的数量(头的数量必须能被维度整除)
            num_layers=layer_num,  # 层数，即最终的Encoder由6个EncoderLayer组成
        )  # 最终transformer_encoder的输入形式是(序列长度10，批大小15，维度50)，输出也还是(序列长度10，批大小15，维度50)，大小都不变
        self.fc = nn.Linear(sequence_length * dimension, out_feature)  # 全连接层，输入是 序列长度10*维度50，输出是2

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 转换维度，原本x_train是(批大小15，序列长度10，维度50)，变为TransformerEncoder需要的(序列长度10，批大小15，维度50)
        x = self.transformer_encoder(x)  # 经过transformer_encoder，x_train的形状还是(序列长度10，批大小15，维度50)
        x = x.permute(1, 0, 2)  # 恢复原始维度顺序，准备输入全连接
        x = x.flatten(1)  # 拉平，x维度是(批大小15，序列长度10，维度50)，指定维度1，拉平为(15,500)
        x = self.fc(x)  # 经过全连接
        x = softmax(x, dim=1)  # softmax
        return x


def classify_dataset():
    # 区分数据集和训练集
    train_sample_list = []
    test_sample_list = []
    namespace_list = os.listdir(Config.dataset_dir)
    for namespace in namespace_list:
        chaos_list = os.listdir(os.path.join(Config.dataset_dir, namespace))
        for chaos in chaos_list:
            file_name_list = os.listdir(os.path.join(Config.dataset_dir, namespace, chaos))
            sample_list = []
            for file_name in file_name_list:
                sample_list.append(os.path.join(Config.dataset_dir, namespace, chaos, file_name))
            test_num = round(0.2 * len(sample_list))
            test_list = random.sample(sample_list, test_num)
            for test_name in test_list:
                sample_list.remove(test_name)
            train_sample_list.extend(sample_list)
            test_sample_list.extend(test_list)

    return train_sample_list, test_sample_list


# 加载样本，并对数据集进行编码
def load_sample(sample):
    sample_encode_sequence = []
    trace_class_data = data_preprocessing(sample)
    # 排除固定延时影响
    trace_class_data = remove_region_latency(trace_class_data)
    # 加载编码数据
    for trace in trace_class_data:
        encode_list = trace.trace_encode()
        length = len(encode_list)
        for i in range(8 - length):
            encode_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        sample_encode_sequence.append(encode_list)
    # 加载标签
    label_index = get_label_index(sample)
    label_sequence = [label_index] * len(sample_encode_sequence)

    return sample_encode_sequence, label_sequence


def load_dataset(sample_list):
    dataset_list = []
    label_list = []
    for sample in sample_list:
        sample_encode_sequence, label_sequence = load_sample(sample)
        dataset_list.extend(sample_encode_sequence)
        label_list.extend(label_sequence)
    dataset = MyDataSet(dataset_list, label_list)
    return dataset


# 计算得到标签的索引值
def get_label_index(sample: str):
    sample_path_list = sample.split('\\')
    namespace = sample_path_list[-3]
    sample_name = sample_path_list[-2]
    svc = sample_name.split('_')[0]
    region = sample_name.split('_')[1]
    chaos = sample_name.split('_')[2]

    label_index = Config.svc_list.index(svc + '.' + namespace) * len(Config.region_list) * len(Config.chaos_list) \
                  + Config.region_list.index(region) * len(Config.chaos_list) \
                  + Config.chaos_list.index(chaos)
    return label_index


# 样本地址转化为标签名
def sample2label_name(sample: str):
    sample_path_list = sample.split('\\')
    namespace = sample_path_list[-3]
    sample_name = sample_path_list[-2]
    svc = sample_name.split('_')[0]
    region = sample_name.split('_')[1]
    chaos = sample_name.split('_')[2]

    return svc + '.' + namespace + '_' + region + '_' + chaos


def get_label_name(label_index: int):
    svc_index = int(label_index / (len(Config.region_list) * len(Config.chaos_list)))
    remainder = (label_index % (len(Config.region_list) * len(Config.chaos_list)))
    region_index = int(remainder / len(Config.chaos_list))
    chaos_index = remainder % len(Config.chaos_list)
    return Config.svc_list[svc_index] + '_' + Config.region_list[region_index] + '_' + Config.chaos_list[chaos_index]


# 训练模型
def train_model(train_sample_list):
    # 2. 对trace数据进行编码
    train_dataset = load_dataset(train_sample_list)
    train_dataloader = DataLoader(train_dataset, batch_size=15, shuffle=True, num_workers=6)
    # 查看一批数据的格式
    batch_train_data, batch_train_label = next(iter(train_dataloader))
    print("batch shape: ")
    print(batch_train_data.shape, batch_train_data.shape)  # (批大小，序列长度，维度)

    # 判断cuda能否使用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # 4. 构建训练模型
    head_num = 3  # 多头注意力数量
    layer_num = 6  # 层数
    epoch_num = 10  # 迭代次数
    dimension = batch_train_data.shape[-1]
    sequence_length = batch_train_data.shape[1]
    out_feature = len(Config.svc_list) * len(Config.region_list) * len(Config.chaos_list)

    model = TransformerEncoderClassification(dimension, head_num, layer_num, sequence_length, out_feature)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # 优化器


    start_time = time.time()
    print(f'head number = {head_num}, layer number = {layer_num}, out feature number = {out_feature}')
    print(start_time, 'start training')
    print('---------------------------------------------------------------------------------')

    # 训练模型
    train_loss = 0
    for epoch in range(epoch_num):
        for i, (batch_train_data, batch_train_label) in enumerate(train_dataloader):
            batch_train_data = batch_train_data
            batch_train_label = batch_train_label
            outputs = model(batch_train_data)
            # 计算误差
            loss = criterion(outputs, batch_train_label.long())
            # 计算准确率
            _, label_pred = torch.max(outputs.data, dim=1)
            # 清空上一次梯度
            optimizer.zero_grad()
            # 反向传播、更新参数
            loss.backward()
            optimizer.step()
            # 计算训练集acc、loss并输出
            train_acc = (label_pred == batch_train_label).sum() / len(batch_train_label)
            train_loss += loss.item()
            if (i + 1) % 10 == 0:  # 每10次迭代输出一次
                print('[%d %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, train_loss / 10, train_acc))
                train_loss = 0.0

    end_time = time.time()
    print(end_time, 'end training')
    print("training time: ", end_time - start_time)
    print('---------------------------------------------------------------------------------')

    # 保存模型，之后利用
    torch.save(model, 'model.pt')


def test_model(test_sample_list):
    # 加载模型
    model = torch.load('model.pt')

    # 判断cuda能否使用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 6. 测试
    localization_list = []  # 每个样本根因定位的排名
    for sample in test_sample_list:
        # 获得trace数据，并进行异常检测
        test_trace_class_data = data_preprocessing(sample)
        result = anomaly_detect(test_trace_class_data, 1)
        if result is True:
            # 触发根因定位程序
            sample_encode_sequence, label_sequence = load_sample(sample)
            sample_encode_sequence = torch.Tensor(sample_encode_sequence).to(device)
            label_sequence = torch.Tensor(label_sequence).to(device)
            outputs = model(sample_encode_sequence)
            # 将每个trace的结果相加
            result = torch.sum(outputs, dim=0)
            result_sorted, result_sorted_index = torch.sort(result, dim=0, descending=True)
            result_sorted_index = result_sorted_index.numpy().tolist()
            rank_list = []  # 根因排名列表
            for result_index in result_sorted_index:
                rank_list.append(get_label_name(result_index))
            print('rank list:', rank_list)
            label_name = sample2label_name(sample)
            localization_list.append(rank_list.index(label_name))
    rank_1_pred = np.sum(list(map(lambda x: x < 1, localization_list))) / float(len(localization_list))
    rank_3_pred = np.sum(list(map(lambda x: x < 3, localization_list))) / float(len(localization_list))
    rank_5_pred = np.sum(list(map(lambda x: x < 5, localization_list))) / float(len(localization_list))
    print("rank 1 prediction: ", rank_1_pred)
    print("rank 3 prediction: ", rank_3_pred)
    print("rank 5 prediction: ", rank_5_pred)



if __name__ == '__main__':
    # 1. 加载数据集，并区分训练集和测试集（如何分，随机分还是按照故障分）
    train_sample_list, test_sample_list = classify_dataset()
    train_model(train_sample_list)
    # test_model(test_sample_list)











