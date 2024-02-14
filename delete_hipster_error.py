from pickle_utils import trace_load, pickle_save
import os
import shutil

namespace_list = os.listdir('dataset')
for namespace in namespace_list:
    chaos_list = os.listdir(os.path.join('dataset', namespace))
    for chaos in chaos_list:
        sample_list = os.listdir(os.path.join('dataset', namespace, chaos))
        for sample in sample_list:
            dir_path = os.path.join('dataset', namespace, chaos, sample)
            # 去除hipster1和2中有问题的trace
            save_trace_list = []
            traces = trace_load(os.path.join(dir_path, 'hipster', 'normal.pkl'))
            for trace in traces:
                is_anomaly = False
                calls = trace['call']
                for call in calls:
                    if 'recommendationservice' in call:
                        is_anomaly = True
                    if 'OTHER_SYSTEM' in call:
                        is_anomaly = True
                if is_anomaly is False:
                    save_trace_list.append(trace)

            pickle_save(save_trace_list, os.path.join(dir_path, 'hipster', 'normal.pkl'))

            traces = trace_load(os.path.join(dir_path, 'hipster2', 'normal.pkl'))
            for trace in traces:
                is_anomaly = False
                calls = trace['call']
                for call in calls:
                    if 'recommendationservice' in call:
                        is_anomaly = True
                    if 'OTHER_SYSTEM' in call:
                        is_anomaly = True
                if is_anomaly is False:
                    save_trace_list.append(trace)

            pickle_save(save_trace_list, os.path.join(dir_path, 'hipster2', 'normal.pkl'))
