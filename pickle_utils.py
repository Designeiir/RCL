import pickle


# 加载trace数据
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


# 将字典存进pkl文件中
def pickle_save(data, save_path):
    with open(save_path, 'wb') as fw:
        pickle.dump(data, fw)


if __name__ == '__main__':
    load_path = 'dataset/bookinfo/details-v1_cloud_mem/cloud_pod_details-v1_details-v1-5c6887cb89-pw9p2_mem_3/hipster2/normal.pkl'
    # dicts = dict_load(load_path)
    #     for key in dicts.keys():
    #         print(key,  ':  ', dicts[key])

    traces = trace_load(load_path)
    for trace in traces:
        print(trace)


