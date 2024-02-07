import os
import pickle
import numpy as np
from config import Config
from data_preprocessing import Trace, Span

# 加载正常情况下的pod延时和net延时，并将其排序合并
# 加载字典数据
def dict_load(path):
    data = {}
    with open(path, 'rb') as f:
        while True:
            try:
                tmp_dict = pickle.load(f)
                for key in tmp_dict.keys():
                    if key not in data.keys():
                        data[key] = []
                    value = tmp_dict[key]
                    data[key] = data[key] + value
            except EOFError:
                break
    return data


def trace_load(file_path: str):
    data = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                data.extend(pickle.load(f))
            except EOFError:
                break
    return data


# 对原来的调用名进行处理，将其变为SVC.REGION的格式
def call_process(span_call: str):
    regions = Config.regions
    # 获得服务名称
    pod_name = span_call.split('.')[0]
    pod_name_list = pod_name.split('-')
    deployment_name = pod_name.replace('-' + pod_name_list[-2] + '-' + pod_name_list[-1], '')
    svc_name = deployment_name.replace('-edge', '')
    # 获得pod所在网段
    region = ''
    pod_ip = span_call.split('-')[-1]
    pod_ip_list = pod_ip.split('.')
    pod_ip_pre = pod_ip_list[0] + '.' + pod_ip_list[1] + '.' + pod_ip_list[2] + '.'
    for key in regions.keys():
        value = regions[key]
        for region_pre in value:
            if pod_ip_pre == region_pre:
                region = key
    # 返回调用信息
    return svc_name + '.' + region


def load_latency_data():
    # 加载正常情况下的pod延时和net延时，并将其排序合并
    normal_dir = 'normal'
    paths = os.listdir(normal_dir)
    namespaces = Config.namespaces

    request_latency_dict = {}
    response_latency_dict = {}
    process_time_dict = {}

    # 遍历每个文件夹下的所有命名空间
    for path in paths:
        for namespace in namespaces:
            net_path = normal_dir + '/' + path + '/' + namespace + '/trace_net_latency.pkl'
            net_data = dict_load(net_path)

            # 对net_data分开处理
            for key in net_data.keys():
                # 不对OTHER_NODE进行处理
                if key.find('OTHER_NODE') >= 0:
                    continue
                caller = key.split('&')[0]
                callee = key.split('&')[1]
                # 将podName处理为调用对
                call_region_pair = (call_process(caller), call_process(callee))
                if call_region_pair not in request_latency_dict.keys():
                    request_latency_dict[call_region_pair] = []
                if call_region_pair not in response_latency_dict.keys():
                    response_latency_dict[call_region_pair] = []
                # 分别将数据加入到两个dict中
                for net_latency in net_data[key]:
                    # 特殊处理
                    if net_latency[0] > 35000 or net_latency[1] > 35000:
                        continue
                    request_latency_dict[call_region_pair].append(net_latency[0])
                    response_latency_dict[call_region_pair].append(net_latency[1])

            # 处理process_time
            process_path = normal_dir + '/' + path + '/' + namespace + '/normal.pkl'
            trace_list = trace_load(process_path)
            for trace_dict in trace_list:
                trace_class = Trace()
                result = trace_class.trace_doc2class(trace_dict)
                if result is False:
                    continue
                for span in trace_class.spans:
                    process_time = span.process_time
                    call_region_pair = (span.caller_svc.split('.')[0] + '.' + span.caller_region, span.callee_svc.split('.')[0] + '.' + span.callee_region)
                    if call_region_pair not in process_time_dict.keys():
                        process_time_dict[call_region_pair] = []
                    process_time_dict[call_region_pair].append(process_time)


    return request_latency_dict, response_latency_dict, process_time_dict

# 对数据进行排序以及删除
def latency_process(latency_dict):
    remove_rate = 0.05
    # 对每个调用对进行排序
    for key in latency_dict:
        latency_list = latency_dict[key]
        latency_list.sort()
        # 删除每个list最小的和最大的一部分
        remove_size = int(len(latency_list) * remove_rate)
        for i in range(remove_size):
            # 删除元素的第一个和最后一个
            latency_list.pop(0)
            latency_list.pop(-1)

    return latency_dict


# 对跨网段通信进行分类，分为网段内和其他跨网段
def call_region_classify(latency_dict):
    same_region_dict = {}
    cross_region_dict = {}
    for key in latency_dict:
        caller_region = key[0].split('.')[1]
        callee_region = key[1].split('.')[1]
        if caller_region == callee_region:
            same_region_dict[key] = latency_dict[key]
        else:
            cross_region_dict[key] = latency_dict[key]

    return same_region_dict, cross_region_dict

# 获得这场状态下服务调用的延时信息
def get_normal_latency(request_same_region_dict, response_same_region_dict, process_time_dict):
    normal_latency_dict = {}
    for key in request_same_region_dict.keys():
        # 获取调用的服务名
        request_list = request_same_region_dict[key]
        response_list = response_same_region_dict[key]
        process_list = process_time_dict[key]
        # 获得平均值和标准差
        request_mean = round(np.mean(request_list), 2)
        request_std = round(np.std(request_list), 2)
        response_mean = round(np.mean(response_list), 2)
        response_std = round(np.std(response_list), 2)
        process_mean = round(np.mean(process_list), 2)
        process_std = round(np.std(process_list), 2)
        # 获取调用服务对
        svc_pair = (key[0].split('.')[0], key[1].split('.')[0])
        normal_latency_dict[svc_pair] = [request_mean, request_std, process_mean, process_std, response_mean, response_std]
    return normal_latency_dict


# 获得各个网段的延时平均值
def get_region_latency(request_cross_region_dict, response_cross_region_dict, normal_latency_dict):
    region_latency_dict = {}
    region_latency_statistic_dict = {}

    for key in request_cross_region_dict.keys():
        # 获取调用的服务名和网段名
        svc_pair = (key[0].split('.')[0], key[1].split('.')[0])
        region_pair = (key[0].split('.')[1], key[1].split('.')[1])
        # 更换顺序
        if region_pair not in Config.cross_region_pair:
            region_pair = (key[1].split('.')[1], key[0].split('.')[1])
        if svc_pair in normal_latency_dict.keys():
            # 获得网段内的延时平均值
            request_mean = normal_latency_dict[svc_pair][0]
            response_mean = normal_latency_dict[svc_pair][4]
            # 跨网段延时减去网段内通信平均值
            request_cross_region_dict[key] = [i - request_mean for i in request_cross_region_dict[key]]
            response_cross_region_dict[key] = [i - response_mean for i in response_cross_region_dict[key]]
            # 查看网段key是否存在
            if region_pair not in region_latency_dict.keys():
                region_latency_dict[region_pair] = []
                region_latency_dict[region_pair] = (request_cross_region_dict[key], response_cross_region_dict[key])
            else:
                region_latency_dict[region_pair] = (region_latency_dict[region_pair][0] + request_cross_region_dict[key], region_latency_dict[region_pair][1] + response_cross_region_dict[key])
        else:
            continue
    # 求出平均值和标准差
    for key in region_latency_dict:
        request_list = region_latency_dict[key][0]
        response_list = region_latency_dict[key][1]
        request_mean = round(np.mean(request_list), 2)
        request_std = round(np.std(request_list), 2)
        response_mean = round(np.mean(response_list), 2)
        response_std = round(np.std(response_list), 2)
        region_latency_statistic_dict[key] = [request_mean, request_std, response_mean, response_std]
    return region_latency_statistic_dict

# 将字典存进pkl文件中
def dict_save(data, save_path):
    with open(save_path, 'wb') as fw:
        pickle.dump(data, fw)


if __name__ == '__main__':
    # 加载数据
    request_latency_dict, response_latency_dict, process_time_dict = load_latency_data()
    # 数据处理
    request_latency_dict = latency_process(request_latency_dict)
    response_latency_dict = latency_process(response_latency_dict)
    process_time_dict = latency_process(process_time_dict)
    # 根据网段分类
    request_same_region_dict, request_cross_region_dict = call_region_classify(request_latency_dict)
    response_same_region_dict, response_cross_region_dict = call_region_classify(response_latency_dict)
    # 得到各个调用在网段内通信的平均值和标准差，存在某个位置
    normal_latency_dict = get_normal_latency(request_same_region_dict, response_same_region_dict, process_time_dict)
    # 求出各个网段的通信的平均值和标准差，存在某个位置
    region_latency_statistic_dict = get_region_latency(request_cross_region_dict, response_cross_region_dict, normal_latency_dict)
    # 将信息存进文件中
    dict_save(normal_latency_dict, 'normal_latency.pkl')
    dict_save(region_latency_statistic_dict, 'region_latency.pkl')

