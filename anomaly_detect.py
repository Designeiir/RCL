from config import Config

# 对数据进行异常检测，返回一定时间内的trace数据
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