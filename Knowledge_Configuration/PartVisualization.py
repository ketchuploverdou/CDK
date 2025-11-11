import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# 随机从训练集中选取1000（太多了networkx可能报错）条三元组，查看图的大体情况
# 加载关系字典和实体字典
def get_dict():
    rel_dict = dict()
    ent_dict = dict()
    with open('./data/OpenBG500/relations.dict', 'r') as f:
        lines = f.readlines()
        for line in lines:
            k, v = line.strip().split('\t')
            rel_dict[k] = v
    with open('./data/OpenBG500/entities.dict', 'r') as f:
        lines = f.readlines()
        for line in lines:
            k, v = line.strip().split('\t')
            ent_dict[k] = v
    return rel_dict, ent_dict


rel_dict, ent_dict = get_dict()
# 选取一部分数据
sample_num = 100
s = np.random.randint(0, 1242550 - sample_num)
# data = []
with open('./data/OpenBG500/train.txt', 'r') as f:
    data = f.readlines()
    data = data[s:s + sample_num]


# 实体和关系转换为ids
def to_ids(data, rel_dict, ent_dict):
    data = data.strip().split('\t')
    return ent_dict[data[0]], ent_dict[data[2]], rel_dict[data[1]]


data = [to_ids(i, rel_dict, ent_dict) for i in data]
plt.figure(figsize=(10, 10), dpi=200)
graph = nx.Graph()
graph.add_weighted_edges_from(data)
node_color = np.concatenate([np.linspace(0, 1, sample_num)[:, None], np.zeros([sample_num, 2])], axis=1)
# 绘图的配置
options = {
    'node_color': 'black',
    'node_size': 10,
    'width': 3,
    # 'node_color': node_color,
    'width': 0.5
}
nx.draw_networkx(graph, **options)
plt.savefig('./result/images/graph.png')
plt.show()
