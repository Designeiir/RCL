import pickle
from config import Config
import sys
from pickle_utils import trace_load, dict_load
from data_structure import Trace, Span_Pair
from anomaly_detect import anomaly_detect
import os
import random


# 加载数据并进行数据预处理
def data_preprocessing(dir_path: str):
    # 加载某个文件夹中的所有trace
    trace_data = []
    trace_class_data = []
    for namespace in Config.namespaces:
        normal_trace_data = trace_load(dir_path + '/' + namespace + '/' + 'normal.pkl')
        abnormal_trace_data = trace_load(dir_path + '/' + namespace + '/' + 'abnormal.pkl')
        trace_data.extend(normal_trace_data)
        trace_data.extend(abnormal_trace_data)

    # 排除trace数据中重复元素
    # trace_data = list(set(trace_data))
    # 将其转变为class
    for trace_dict in trace_data:
        trace_class = Trace()
        trace_class.trace_doc2class(trace_dict)
        trace_class_data.append(trace_class)
    # 对trace数据进行排序
    trace_class_data.sort()

    return trace_class_data


# 排除trace跨网段数据影响
def remove_region_latency(trace_class_data):
    if Config.normal_latency_dict == {}:
        Config.load_latency_dict()
    for trace in trace_class_data:
        for span in trace.spans:
            if span.caller_region == span.callee_region:
                continue
            if span.caller_svc == 'OTHER_SVC' or span.callee_svc == 'OTHER_SVC':
                continue
            call_region_pair = (span.caller_region, span.callee_region)
            if call_region_pair not in Config.cross_region_pair:
                call_region_pair = (span.callee_region, span.caller_region)
            # 将请求时间和响应时间减去跨网段通信延时
            span.request_latency = max(span.request_latency - Config.region_latency_dict[call_region_pair][0], 1)
            span.response_latency = max(span.response_latency - Config.region_latency_dict[call_region_pair][2], 1)

    return trace_class_data


# 区分数据集和训练集
def classify_dataset():
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


if __name__ == '__main__':
    # dir_path = 'bookinfo-details-cpu-1'
    # trace_class_data = data_preprocessing(dir_path)
    # print(trace_class_data)

    dir_path = 'dataset/bookinfo/details-v1_cloud_net/cloud_pod_details-v1_details-v1-66857885-k4ngb_net_1'
    trace_class_data = data_preprocessing(dir_path)
    anomaly_detect(trace_class_data, 1)

