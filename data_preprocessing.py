import pickle
from config import Config
import sys

# span类
class Span(object):
    def __init__(self):
        self.position = None
        self.caller_svc = None
        self.callee_svc = None
        self.caller_region = None
        self.callee_region = None
        self.caller_http_code = None
        self.callee_http_code = None
        self.request_latency = None
        self.process_time = None
        self.response_latency = None
        self.duration = None
        self.is_anomaly = False


    def __str__(self):
        return f'(Anomaly: {self.is_anomaly} Call: ({self.caller_svc}.{self.caller_region}, {self.callee_svc}.{self.callee_region}), Latency: [{self.request_latency}, {self.process_time}, {self.response_latency}], Http_code: [{self.caller_http_code}, {self.callee_http_code}])'


    def get_anomaly(self):
        # 通过http状态码判断是否异常
        if self.caller_http_code >= 300 or self.caller_http_code < 200:
            self.is_anomaly = True
            return True
        if self.callee_http_code >= 300 or self.callee_http_code < 200:
            self.is_anomaly = True
            return True
        if self.caller_svc == 'OTHER_SVC' or self.callee_svc == 'OTHER_SVC':
            return False
        # 通过延时信息判断是否异常
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
            # 将请求时间和响应时间减去跨网段通信延时
            request_latency_without_region = self.request_latency - Config.region_latency_dict[call_region_pair][0]
            response_latency_without_region = self.response_latency - Config.region_latency_dict[call_region_pair][2]
        # 获得服务调用的历史数据
        request_mean = Config.normal_latency_dict[call_svc_pair][0]
        request_std = Config.normal_latency_dict[call_svc_pair][1]
        process_mean = Config.normal_latency_dict[call_svc_pair][2]
        process_std = Config.normal_latency_dict[call_svc_pair][3]
        response_mean = Config.normal_latency_dict[call_svc_pair][4]
        response_std = Config.normal_latency_dict[call_svc_pair][5]
        # 判断三个时间是否超过范围
        if request_latency_without_region > request_mean + 3 * request_std and request_latency_without_region > 2000:
            self.is_anomaly = True
            return True
        if response_latency_without_region > response_mean + 3 * response_std and request_latency_without_region > 2000:
            self.is_anomaly = True
            return True
        if self.process_time > process_mean + 3 * process_std:
            self.is_anomaly = True
            return True
        self.is_anomaly = False
        return False

    def span_encode(self):
        if self.caller_svc == 'OTHER_SVC':
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
        return [caller_svc_index, caller_region_index, self.caller_http_code, callee_svc_index, callee_region_index, self.callee_http_code, self.request_latency, self.process_time, self.response_latency]


# trace类
class Trace(object):
    def __init__(self):
        self.spans = []
        self.timestamp = 0
        self.trace_dict = None
        self.is_anomaly = None

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

    # 将trace文本转化为Trace类，用Span类进行填充
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
            span = Span()
            span.position = i
            # 获取span的调用信息
            caller = spans_call[i][0]
            callee = spans_call[i][1]
            if 'OTHER_NODE' == caller:
                span.caller_svc = 'OTHER_SVC'
                span.caller_region = 'OTHER_REGION'
                span.callee_svc, span.callee_region = span_call_process(callee)
            elif 'OTHER_NODE' == callee:
                span.callee_svc = 'OTHER_SVC'
                span.callee_region = 'OTHER_REGION'
                span.caller_svc, span.caller_region = span_call_process(caller)
            else:
                span.caller_svc, span.caller_region = span_call_process(caller)
                span.callee_svc, span.callee_region = span_call_process(callee)
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
                if other_span.caller_svc == callee_svc and other_span.caller_region == callee_region:
                    span.process_time -= other_span.duration


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




# 定义不同网段
regions = {'cloud': ['10.244.13.', '10.244.0.', '10.244.5.', '10.244.6.'],
          'edge-1': ['10.244.8.', '10.244.12.'],
          'edge-2': ['10.244.9.', '10.244.11.']}


# 对span的调用关系进行处理，得到服务名，命名空间和网段名称
def span_call_process(span_call: str):
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
    for key in regions.keys():
        value = regions[key]
        for region_pre in value:
            if pod_ip_pre == region_pre:
                region = key
    # 返回调用信息
    return svc_name + '.' + namespace, region

# 加载Trace数据
def trace_load(file_path: str):
    data = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                data.extend(pickle.load(f))
            except EOFError:
                break
    return data


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


# 排除trace数据影响
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


def anomaly_detect(trace_class_data, abnormal_number):
    # 判断Config中是否加载历史调用数据
    if Config.normal_latency_dict == {}:
        Config.load_latency_dict()
    normal_count = 0
    abnormal_count = 0
    is_anomaly = False
    # 得到该样本trace的开始时间和结束时间
    start_timestamp = trace_class_data[0].timestamp
    end_timestamp = trace_class_data[-1].timestamp
    detect_duration = 60 * 1000 * 1000  # 时间窗口为一分钟
    forward_duration = 2 * 60 * 1000 * 1000
    backward_duration = 5 * 60 * 1000 * 1000
    # 从开始时间开始，加载一个时间窗口的trace
    while True:
        count = 0
        if end_timestamp < start_timestamp:
            break
        window_traces = get_time_window_traces(trace_class_data, start_timestamp, 0, detect_duration)
        # 对每条trace数据，判断是否异常
        for trace in window_traces:
            result = trace.get_anomaly()
            if result is True:
                abnormal_count += 1
                count += 1
            else:
                normal_count = normal_count + 1
        # 当异常trace数量达到一定时，判断为异常(统计时要注释掉)
        if count >= abnormal_number:
            # 获取一个更大的时间窗口
            return get_time_window_traces(trace_class_data, start_timestamp, forward_duration, backward_duration)
        start_timestamp += detect_duration

    print("normal :", normal_count, "abnormal :", abnormal_count)
    return None


# 获取一个时间窗口内的trace
def get_time_window_traces(trace_class_data, start_time, forward_duration, backward_duration):
    time_window_traces = []
    for trace in trace_class_data:
        # trace在时间窗口之间
        if trace.timestamp >= start_time - forward_duration and trace.timestamp <= start_time + backward_duration:
            time_window_traces.append(trace)
    return time_window_traces



if __name__ == '__main__':
    # dir_path = 'bookinfo-details-cpu-1'
    # trace_class_data = data_preprocessing(dir_path)
    # print(trace_class_data)

    dir_path = 'dataset/bookinfo/details-v1_cloud_net/cloud_pod_details-v1_details-v1-66857885-k4ngb_net_1'
    trace_class_data = data_preprocessing(dir_path)
    anomaly_detect(trace_class_data, 1)

