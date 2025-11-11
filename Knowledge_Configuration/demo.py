import pgl
import os
import paddle
from reader import read_trigraph
import paddle.distributed as dist
from model import KGEModel


class Args:
    def __init__(self):
        self.seed = 0
        self.data_path = './data/'
        self.save_path = './result/checkpoint/transe/transe_data2023_d_400_g_19.9_e_gpu_r_gpu_l_Logsigmoid_lr_0.01_0.1_KGE'
        self.init_from_ckpt = './result/checkpoint/transe/transe_data2023_d_400_g_19.9_e_gpu_r_gpu_l_Logsigmoid_lr_0.01_0.1_KGE'
        self.data_name = 'data2023'
        self.use_dict = False
        self.kv_mode = 'kv'
        self.valid_percent = 1.
        self.filter_sample = False
        self.filter_eval = True
        self.weighted_loss = False
        self.model_name = 'transe'
        self.ent_dim = 400
        self.rel_dim = 400
        self.ent_emb_on_cpu = False
        self.rel_emb_on_cpu = False
        self.num_chunks = 5
        self.cpu_lr = 0.1
        self.mix_cpu_gpu = False
        self.gamma = 19.9
        self.ote_size = 0
        self.ote_scale = 1
        self.use_feature = False
        self.use_dict = False


def build_model(args):
    trigraph = read_trigraph(args.data_path, args.data_name, args.use_dict,
                             args.kv_mode)
    if args.valid_percent < 1:
        trigraph.sampled_subgraph(args.valid_percent, dataset='valid')

    use_filter_set = args.filter_sample or args.filter_eval or args.weighted_loss
    if use_filter_set:
        filter_dict = {
            'head': trigraph.true_heads_for_tail_rel,
            'tail': trigraph.true_tails_for_head_rel
        }
    else:
        filter_dict = None

    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    model = KGEModel(args.model_name, trigraph, args)
    if args.init_from_ckpt:
        state_dict = paddle.load(
            os.path.join(args.init_from_ckpt, 'params.pdparams'))
    return model


def get_dict():
    rel_dict = dict()
    ent_dict = dict()
    with open('./data/data2023/relations.dict', 'r') as f:
        lines = f.readlines()
        for line in lines:
            k, v = line.strip().split('\t')
            rel_dict[k] = v
    with open('./data/data2023/entities.dict', 'r') as f:
        lines = f.readlines()
        for line in lines:
            k, v = line.strip().split('\t')
            ent_dict[k] = v
    return rel_dict, ent_dict


def do_predict(model, head, relation):
    model.eval()
    rel_dict, ent_dict = get_dict()
    h = paddle.to_tensor([ent_dict[head]], 'int64')
    r = paddle.to_tensor([rel_dict[relation]], 'int64')
    with paddle.no_grad():
        t_score = model.predict(h, r, mode='tail')
    t_score = t_score.argsort(descending=True)
    return t_score[:, :10][0]


def get_key(id_):
    _, ent_dict = get_dict()
    for item in ent_dict.items():
        if str(id_) in item:
            return item[0]


# ===========示例==============
args = Args()
model = build_model(args)
head = 'kp10'
relation = 'rel_06'
tails = do_predict(model, head, relation)
print(f"得分最高的前十个实体id为： {tails.numpy()}")
print(f"{head} - {relation} --> {get_key(int(tails[0]))}")
