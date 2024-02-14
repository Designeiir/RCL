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

