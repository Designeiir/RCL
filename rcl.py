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
import wandb

class MyDataSet(Dataset):  # 定义类，用于构建数据集
    def __init__(self, data, label, mask):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.mask[index]

    def __len__(self):
        return self.length


class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model=9, dropout=0.1, max_len=8):
        # d_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0), torch.arange(0, d_model, 2).float() / d_model)
        div_term1 = torch.pow(torch.tensor(10000.0), torch.arange(1, d_model, 2).float() / d_model)
        # 高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        # 这里是为了与x的维度保持一致，释放了一个维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        temp_pe = self.pe.repeat(x.size(0), 1, 1)
        x = x + temp_pe
        return x

    def add_one(self, x):
        temp_pe = self.pe.squeeze(0)
        x = x + temp_pe
        return x


class TransformerEncoderClassification(nn.Module):
    def __init__(self, dimension, head_num, layer_num, sequence_length, out_feature):
        super(TransformerEncoderClassification, self).__init__()
        # 先用nn.TransformerEncoderLayer构造Encoder层，再用nn.TransformerEncoder构造最终的Encoder，最终的Encoder可以包括几个层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dimension, nhead=head_num),  # 两个参数分别是输入样本的维度(50)和Encoder头的数量(头的数量必须能被维度整除)
            num_layers=layer_num,  # 层数，即最终的Encoder由6个EncoderLayer组成
            enable_nested_tensor=True,
        )  # 最终transformer_encoder的输入形式是(序列长度10，批大小15，维度50)，输出也还是(序列长度10，批大小15，维度50)，大小都不变
        self.fc = nn.Linear(sequence_length * dimension, out_feature)  # 全连接层，输入是 序列长度10*维度50，输出是2

    def forward(self, x, mask_matrix):
        x = x.permute(1, 0, 2)  # 转换维度，原本x_train是(批大小15，序列长度10，维度50)，变为TransformerEncoder需要的(序列长度10，批大小15，维度50)
        x = self.transformer_encoder(x, src_key_padding_mask=mask_matrix)  # 经过transformer_encoder，x_train的形状还是(序列长度10，批大小15，维度50)
        x = x.permute(1, 0, 2)  # 恢复原始维度顺序，准备输入全连接
        x = x.flatten(1)  # 拉平，x维度是(批大小15，序列长度10，维度50)，指定维度1，拉平为(15,500)
        x = self.fc(x)  # 经过全连接
        # x = softmax(x, dim=1)  # softmax
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
    mask_sequence = []
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
        mask_sequence.append(get_mask(length, 8))
    # 加载标签
    label_index = get_label_index(sample)
    label_sequence = [label_index] * len(sample_encode_sequence)

    return sample_encode_sequence, label_sequence, mask_sequence


def get_mask(encode_length, max_length):
    mask = []
    for i in range(max_length):
        if i < encode_length:
            mask.append(False)
        else:
            mask.append(True)

    return mask

def load_dataset(sample_list):
    dataset_list = []
    label_list = []
    mask_list = []
    for sample in sample_list:
        sample_encode_sequence, label_sequence, mask_sequence = load_sample(sample)
        dataset_list.extend(sample_encode_sequence)
        label_list.extend(label_sequence)
        mask_list.extend(mask_sequence)
    dataset = MyDataSet(dataset_list, label_list, mask_list)
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
    # login
    wandb.login(key='968d99f8b316c7232ecd8fef76d74aac3ef5c54c')
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="rcl-transformer-encoder",

        # track hyperparameters and run metadata
        config={
            "architecture": "transformer-encoder",
            "dataset": "bookinfo",
            "epochs": 50,
            "multi-head": 3,
            "layer": 6,
            "batch-size": 256,
            "optimizer": "Adam"
        }
    )

    # 2. 对trace数据进行编码
    train_dataset = load_dataset(train_sample_list)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6)
    # 查看一批数据的格式
    batch_train_data, batch_train_label, batch_mask = next(iter(train_dataloader))
    print("batch shape: ")
    print(batch_train_data.shape, batch_train_data.shape, batch_mask)  # (批大小，序列长度，维度)

    # 判断cuda能否使用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # 4. 构建训练模型
    head_num = Config.head_num
    layer_num = Config.layer_num
    epoch_num = Config.epoch_num
    dimension = batch_train_data.shape[-1]
    sequence_length = batch_train_data.shape[1]
    out_feature = len(Config.svc_list) * len(Config.region_list) * len(Config.chaos_list)

    model = TransformerEncoderClassification(dimension, head_num, layer_num, sequence_length, out_feature).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)  # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    position_encoding = PositionalEncoding(device)


    start_time = time.time()
    print(f'batch size = {Config.batch_size}, optimizer = {optimizer.__class__}, dataset = bookinfo, epoch num = {epoch_num}')
    print(f'head number = {head_num}, layer number = {layer_num}, out feature number = {out_feature}')
    print(start_time, 'start training')
    print('---------------------------------------------------------------------------------')

    # 开始迭代
    for epoch in range(epoch_num):
        train_correct = 0
        train_accuracy = 0
        train_loss = 0
        batches_average_loss = 0
        train_total = 0

        # 对每一个batch进行训练
        for i, (batch_train_data, batch_train_label, batch_mask) in enumerate(train_dataloader):
            batch_train_data = batch_train_data.to(device)
            batch_train_label = batch_train_label.to(device)
            batch_mask = batch_mask.to(device)  # padding mask
            batch_train_data = position_encoding(batch_train_data)
            outputs = model(batch_train_data, batch_mask)
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
            correct = (label_pred == batch_train_label).sum().item()
            train_correct += correct
            accuracy = correct / len(batch_train_label)
            train_loss += loss.item()
            batches_average_loss += loss.item()
            train_total += batch_train_label.size(0)

            if (i + 1) % 10 == 0:  # 每10个batch输出一次
                print('[%d %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, batches_average_loss / 10, accuracy))
                batches_average_loss = 0

        train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        print('[epoch %d] loss: %.3f acc: %.3f' % (epoch + 1, train_loss, train_accuracy))
        # log metrics to wandb
        wandb.log({"acc": train_accuracy, "loss": train_loss})

    end_time = time.time()
    print(f'batch size = {Config.batch_size}, optimizer = {optimizer.__class__}, dataset = bookinfo, epoch num = {epoch_num}')
    print(f'head number = {head_num}, layer number = {layer_num}, out feature number = {out_feature}')
    # 将其写入记录文件中
    with open('result.txt', 'rt') as f:
        f.write(f'batch size = {Config.batch_size}, optimizer = {optimizer.__class__}, dataset = bookinfo, epoch num = {epoch_num}\n')
        f.write(f'head number = {head_num}, layer number = {layer_num}, out feature number = {out_feature}\n')
    print(end_time, 'end training')
    print("training time: %.3fs" % (end_time - start_time))
    print('---------------------------------------------------------------------------------')

    # 保存模型，之后利用
    torch.save(model, 'model.pt')
    wandb.finish()

def test_model(test_sample_list):
    # 加载模型
    model = torch.load('model.pt')
    abnormal_number = 8  # 时间窗口内的异常数量阈值

    # 判断cuda能否使用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    position_encoding = PositionalEncoding(device)

    # 6. 测试
    localization_list = []  # 每个样本根因定位的排名
    for sample in test_sample_list:
        # 获得trace数据，并进行异常检测
        test_trace_class_data = data_preprocessing(sample)
        window_traces = anomaly_detect(test_trace_class_data, abnormal_number)
        if window_traces is not None:
            # 触发根因定位程序
            sample_encode_sequence, label_sequence, mask_sequence = load_sample(sample)
            sample_encode_sequence = torch.Tensor(sample_encode_sequence).to(device)
            mask_sequence = torch.tensor(mask_sequence, dtype=torch.bool).to(device)  # mask
            sample_encode_sequence = position_encoding.add_one(sample_encode_sequence)  # 加入位置编码
            outputs = model(sample_encode_sequence, mask_sequence)
            outputs = softmax(outputs, dim=1)  # softmax
            # 将每个trace的结果相加
            result = torch.sum(outputs, dim=0)
            result_sorted, result_sorted_index = torch.sort(result, dim=0, descending=True)
            result_sorted_index = result_sorted_index.cpu().numpy().tolist()
            rank_list = []  # 根因排名列表
            for result_index in result_sorted_index:
                rank_list.append(get_label_name(result_index))
            label_name = sample2label_name(sample)
            print('label:',label_name, 'rank list:', rank_list)
            localization_list.append(rank_list.index(label_name))
    rank_1_pred = np.sum(list(map(lambda x: x < 1, localization_list))) / float(len(localization_list))
    rank_3_pred = np.sum(list(map(lambda x: x < 3, localization_list))) / float(len(localization_list))
    rank_5_pred = np.sum(list(map(lambda x: x < 5, localization_list))) / float(len(localization_list))
    print("rank 1 prediction: ", rank_1_pred)
    print("rank 3 prediction: ", rank_3_pred)
    print("rank 5 prediction: ", rank_5_pred)

    with open('result.txt', 'rt') as f:
        f.write("rank 1 prediction: " + rank_1_pred + '\n')
        f.write("rank 3 prediction: " + rank_3_pred + '\n')
        f.write("rank 5 prediction: " + rank_5_pred + '\n')
        f.write("----------------------------------")



if __name__ == '__main__':
    train_sample_list, test_sample_list = classify_dataset()
    train_model(train_sample_list)
    test_model(test_sample_list)


# loss 收敛范围， early stop
# 分文件
# 先跑五百轮。。。看看效果
# 准确率（将云边加上再进行对比），召回率（动机）
# 异常检测，延时归一化  stand scalar
# 将数据mask再进入transformer








