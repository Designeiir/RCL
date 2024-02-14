from config import Config
import sys
from torch.utils.data import Dataset
import torch

# span调用对，表示两个服务之间的调用关系
class Span_Pair(object):
    def __init__(self):
        self.position = None  # 在trace中的位置
        self.caller_svc = None  # 调用者服务
        self.callee_svc = None  # 被调用者服务
        self.caller_region = None  # 调用者所在网段
        self.callee_region = None  # 被调用者所在网段
        self.caller_http_code = None  # 调用者状态码
        self.callee_http_code = None  # 被调用者状态码
        self.request_latency = None  # 请求延迟
        self.process_time = None  # 被调用者业务处理时间
        self.response_latency = None  # 响应延迟
        self.duration = None  # 调用对持续时间
        self.is_anomaly = False  # 是否异常

    def __str__(self):
        return f'(Anomaly: {self.is_anomaly} Call: ({self.caller_svc}.{self.caller_region}, {self.callee_svc}.{self.callee_region}), Latency: [{self.request_latency}, {self.process_time}, {self.response_latency}], Http_code: [{self.caller_http_code}, {self.callee_http_code}])'


    # 根据http_code和延迟信息判断服务调用是否异常
    def get_anomaly(self):
        # 通过http状态码判断是否异常
        if self.caller_http_code >= 300 or self.caller_http_code < 200:
            self.is_anomaly = True
            return True
        if self.callee_http_code >= 300 or self.callee_http_code < 200:
            self.is_anomaly = True
            return True
        # 有OTHER_SVC的调用对状态码为200，认为是正常跨系统调用
        if self.caller_svc == 'OTHER_SVC' or self.callee_svc == 'OTHER_SVC':
            return False
        # 加载历史数据信息
        if Config.normal_latency_dict == {} or Config.region_latency_dict == {}:
            Config.load_latency_dict()
        # 获得该span的调用对和地域对
        call_svc_pair = (self.caller_svc.split('.')[0], self.callee_svc.split('.')[0])
        # 判断是否跨网段
        if self.caller_region == self.callee_region:
            request_latency_without_region = self.request_latency
            response_latency_without_region = self.response_latency
        else:
            # 调用跨网段，抹去跨网段延时
            call_region_pair = (self.caller_region, self.callee_region)
            if call_region_pair not in Config.cross_region_pair:
                call_region_pair = (self.callee_region, self.caller_region)
            request_latency_without_region = self.request_latency - Config.region_latency_dict[call_region_pair][0]
            response_latency_without_region = self.response_latency - Config.region_latency_dict[call_region_pair][2]
        # 获得服务调用的历史数据
        request_mean = Config.normal_latency_dict[call_svc_pair][0]
        request_std = Config.normal_latency_dict[call_svc_pair][1]
        process_mean = Config.normal_latency_dict[call_svc_pair][2]
        process_std = Config.normal_latency_dict[call_svc_pair][3]
        response_mean = Config.normal_latency_dict[call_svc_pair][4]
        response_std = Config.normal_latency_dict[call_svc_pair][5]
        # 判断三个延迟时间是否超过范围
        # TODO: 删掉》2000的硬性要求
        # TODO: 如何更精确的检测出异常
        n = Config.anomaly_n
        if request_latency_without_region > request_mean + n * request_std and request_latency_without_region > 2000:
            self.is_anomaly = True
            return True
        if response_latency_without_region > response_mean + n * response_std and request_latency_without_region > 2000:
            self.is_anomaly = True
            return True
        if self.process_time > process_mean + n * process_std:
            self.is_anomaly = True
            return True
        self.is_anomaly = False
        return False

    '''
        对span_pair进行编码
        [ caller_svc_index, caller_region_index, caller_http_code, callee_svc_index, callee_region_index, callee_http_code, request_latency, process_time, response_latency]
    '''
    def span_encode(self):
        if self.caller_svc == 'OTHER_SVC':
            # OTHER_SVC的index设置为-1
            caller_svc_index = -1
            caller_region_index = -1
            callee_svc_index = Config.svc_list.index(self.callee_svc) + 1
            callee_region_index = Config.region_list.index(self.callee_region) + 1
        elif self.callee_svc == 'OTHER_SVC':
            callee_svc_index = -1
            callee_region_index = -1
            caller_svc_index = Config.svc_list.index(self.caller_svc) + 1
            caller_region_index = Config.region_list.index(self.caller_region) + 1
        else:
            caller_svc_index = Config.svc_list.index(self.caller_svc) + 1
            callee_svc_index = Config.svc_list.index(self.callee_svc) + 1
            caller_region_index = Config.region_list.index(self.caller_region) + 1
            callee_region_index = Config.region_list.index(self.callee_region) + 1
        # 对延迟时间标准化
        # request_normalization, process_normalization, response_normalization = self.latency_normalization()
        # return [caller_svc_index, caller_region_index, self.caller_http_code, callee_svc_index, callee_region_index, self.callee_http_code, request_normalization, process_normalization, response_normalization]
        # 不标准化则直接返回
        return [caller_svc_index, caller_region_index, self.caller_http_code, callee_svc_index, callee_region_index, self.callee_http_code, self.request_latency, self.process_time, self.response_latency]

    # 获得延迟信息的标准化
    # def latency_normalization(self):
    #     # 加载历史数据信息
    #     if Config.normal_latency_dict == {} or Config.region_latency_dict == {}:
    #         Config.load_latency_dict()
    #     # 获得该span的调用对和地域对
    #     call_svc_pair = (self.caller_svc.split('.')[0], self.callee_svc.split('.')[0])
    #     # 判断是否跨网段
    #     if self.caller_region == self.callee_region:
    #         request_latency_without_region = self.request_latency
    #         response_latency_without_region = self.response_latency
    #     else:
    #         # 调用跨网段，抹去跨网段延时
    #         call_region_pair = (self.caller_region, self.callee_region)
    #         if call_region_pair not in Config.cross_region_pair:
    #             call_region_pair = (self.callee_region, self.caller_region)
    #         request_latency_without_region = self.request_latency - Config.region_latency_dict[call_region_pair][0]
    #         response_latency_without_region = self.response_latency - Config.region_latency_dict[call_region_pair][2]
    #
    #     # 标准化
    #     request_normalization = (request_latency_without_region - Config.normal_latency_dict[call_svc_pair][0]) / Config.normal_latency_dict[call_svc_pair][1]
    #     process_normalization = (self.process_time - Config.normal_latency_dict[call_svc_pair][2]) / Config.normal_latency_dict[call_svc_pair][3]
    #     response_normalization = (response_latency_without_region - Config.normal_latency_dict[call_svc_pair][4]) / Config.normal_latency_dict[call_svc_pair][5]
    #
    #     return request_normalization, process_normalization, response_normalization


class Trace(object):
    def __init__(self):
        self.spans = []
        self.timestamp = 0  # trace开始时间
        self.trace_dict = None  # trace对应的文本字符串
        self.is_anomaly = None  # trace是否异常

    def __str__(self):
        span_str = '  '
        for span in self.spans:
            span_str = span_str + '  ' + str(span)
        return str(self.is_anomaly) + span_str

    def __lt__(self, other):
        if self.timestamp == -1:
            return False
        if other.timestamp == -1:
            return True
        else:
            return self.timestamp < other.timestamp

    # 将trace文本转化为Trace类，用Span_Pair进行填充
    def trace_doc2class(self, trace_dict: dict):
        self.trace_dict = trace_dict
        # 得到trace信息
        spans_call = trace_dict['call_instance']
        spans_timestamp = trace_dict['timestamp']
        spans_latency = trace_dict['latency']
        http_status = trace_dict['http_status']
        # 得到trace的时间戳
        self.timestamp = sys.maxsize
        for timestamp in spans_timestamp:
            if timestamp != -1 and timestamp < self.timestamp:
                self.timestamp = timestamp
        # 遍历trace中的调用关系
        for i in range(len(spans_call)):
            span = Span_Pair()
            span.position = i
            # 获取span的调用信息
            caller = spans_call[i][0]
            callee = spans_call[i][1]
            if 'OTHER_NODE' == caller:
                span.caller_svc = 'OTHER_SVC'
                span.caller_region = 'OTHER_REGION'
                span.callee_svc, span.callee_region = self.get_span_call(callee)
            elif 'OTHER_NODE' == callee:
                span.callee_svc = 'OTHER_SVC'
                span.callee_region = 'OTHER_REGION'
                span.caller_svc, span.caller_region = self.get_span_call(caller)
            else:
                span.caller_svc, span.caller_region = self.get_span_call(caller)
                span.callee_svc, span.callee_region = self.get_span_call(callee)
            # 获取span的延时信息
            span.duration = spans_latency[2 * i]
            span.request_latency = spans_timestamp[2 * i + 1] - spans_timestamp[2 * i]
            span.response_latency = spans_timestamp[2 * i] + spans_latency[2 * i] - spans_timestamp[2 * i + 1] - spans_latency[2 * i + 1]
            # 获取span的http状态
            span.caller_http_code = http_status[2 * i]
            span.callee_http_code = http_status[2 * i + 1]
            # 将span添加到属性当中
            self.spans.append(span)
        # 计算各个span的process_time
        self.get_span_process_time()
        return True

    # 获取每个span的process time
    def get_span_process_time(self):
        # 遍历每一个span
        for span in self.spans:
            callee_svc = span.callee_svc
            callee_region = span.callee_region
            span.process_time = span.duration - span.request_latency - span.response_latency
            for other_span in self.spans:
                # 跳过同个span
                if other_span.position == span.position:
                    continue
                # 减去下游调用的时间
                if other_span.caller_svc == callee_svc and other_span.caller_region == callee_region:
                    span.process_time -= other_span.duration

    # 根据trace中的span是否异常判断trace是否异常
    def get_anomaly(self):
        for span in self.spans:
            result = span.get_anomaly()
            if result == True:
                self.is_anomaly = True
                return True
        self.is_anomaly = False
        return False

    # 对trace数据进行编码
    def trace_encode(self):
        encode_list = []
        for span in self.spans:
            encode_list.append(span.span_encode())
        return encode_list

    # 获得span的调用信息
    def get_span_call(self, span_call: str):
        # 获得服务名称
        pod_name = span_call.split('.')[0]
        pod_name_list = pod_name.split('-')
        deployment_name = pod_name.replace('-' + pod_name_list[-2] + '-' + pod_name_list[-1], '')
        svc_name = deployment_name.replace('-edge', '')
        # 获得命名空间名称
        namespace = span_call.split('.')[1].replace('-' + span_call.split('.')[1].split('-')[-1], '')
        # 获得pod所在网段
        region = ''
        pod_ip = span_call.split('-')[-1]
        pod_ip_list = pod_ip.split('.')
        pod_ip_pre = pod_ip_list[0] + '.' + pod_ip_list[1] + '.' + pod_ip_list[2] + '.'
        for key in Config.regions.keys():
            value = Config.regions[key]
            for region_pre in value:
                if pod_ip_pre == region_pre:
                    region = key
        # 返回调用信息
        return svc_name + '.' + namespace, region


# 封装数据集
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



