import pickle
class Config(object):
    # namespaces = ['bookinfo', 'hipster', 'hipster2', 'cloud-sock-shop', 'horsecoder-test']
    namespaces = ['bookinfo']

    regions = {'cloud': ['10.244.13.', '10.244.0.', '10.244.5.', '10.244.6.'],
               'edge-1': ['10.244.8.', '10.244.12.'],
               'edge-2': ['10.244.9.', '10.244.11.']}
    cross_region_pair = [('cloud', 'edge-1'), ('cloud', 'edge-2'), ('edge-1', 'edge-2')]

    # 标签索引按照 svc，region，chaos进行排序
    region_list = ['cloud', 'edge-1', 'edge-2']
    svc_list = ['istio-ingressgateway.istio-system', 'productpage-v1.bookinfo', 'details-v1.bookinfo', 'reviews-v1.bookinfo', 'reviews-v2.bookinfo', 'reviews-v3.bookinfo', 'ratings-v1.bookinfo',
               ]
    chaos_list = ['cpu', 'mem', 'net']

    normal_latency_dict = {}
    region_latency_dict = {}

    dataset_dir = 'dataset'

    head_num = 3  # 多头注意力数量
    layer_num = 6  # 层数
    epoch_num = 10  # 迭代次数
    batch_size = 256  # 批数据数量
    anomaly_n = 3
    max_sequence_length = 8

    @classmethod
    def load_latency_dict(cls):
        normal_path = 'normal_latency.pkl'
        region_path = 'region_latency.pkl'
        with open(normal_path, 'rb') as f:
            Config.normal_latency_dict = pickle.load(f)
        with open(region_path, 'rb') as f:
            Config.region_latency_dict = pickle.load(f)
