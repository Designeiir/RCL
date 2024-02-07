import pickle

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

data = dict_load('region_latency.pkl')
for key in data.keys():
    print(key, ': ',  data[key])

from torch.nn.functional import softmax
# import torch
# import torch.nn.functional as F
#
# data = torch.FloatTensor([[3.0, 8.0, 2.0], [45.0, 5.0, 9.0]])
# print(data)
# print(data.shape)
# print(data.type())
#
# prob = F.softmax(data, dim=1)  # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax
# print(prob)
# print(prob.shape)
# print(prob.type())
#
# prob = torch.sum(prob, dim=0)
# print(prob)
# print(prob.shape)
# print(prob.type())
#
# prob, prob_index = torch.sort(prob, dim=0, descending=True)
# print(prob)
# print(prob.shape)
# print(prob.type())
#
# print(prob_index)
# print(prob_index.shape)
# print(prob_index.type())
