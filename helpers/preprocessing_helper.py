import pandas as pd
import pickle
import os
import torch
import tqdm
from utils.datasets import generate_ts_data2
from utils.parse_xyz import parse_xyz_corpus
from preprocessing import index_split

class preprocess_args:
    def __init__(self):
        self.seed = 42
        self.train = 0.0
        self.valid = 0.0
        self.feat_dict = "data/raw_data/feat_dict.pkl"
        self.save_dir = "data/scratch/random_split_42"
        self.ts_data = "data/raw_data/structures.xyz"
        self.rxn_smarts_file = "data/raw_data/smarts.csv"
        self.ban_index = [20568, 20569, 20580, 20581]

def preprocess(args_instance):
    args = args_instance

    # load source chemical reaction dataset
    # The data are augmented reaction data.
    # The original data is placed in even index and the corresponding augmented data is placed in the next index.
    # By chainging reactants and products, the augmented sample is a reverse reaction of original reaction.

    # transition state geometry data of xyz format.
    xyz_blocks = parse_xyz_corpus(args.ts_data)

    # reaction smarts data of csv format.
    df = pd.read_csv(args.rxn_smarts_file)
    rxn_smarts = df.AAM

    # set index of source data to be excluded
    if args.ban_index[0] != -1:
        ban_index = args.ban_index

    # set feature types
    # if there exist pre-defined feat_dict, load the feat_dict
    if os.path.isfile(args.feat_dict):
        feat_dict = pickle.load(open(args.feat_dict, "rb"))
    else:
        print(args.feat_dict, "is not exist. Use default feat_dict.")
        feat_dict = {
            "GetIsAromatic": {},
            "GetFormalCharge": {},
            "GetHybridization": {},
            "GetTotalNumHs": {},
            "GetTotalValence": {},
            "GetTotalDegree": {},
            "GetChiralTag": {},
            "IsInRing": {},
        }

    # generate torch_geometric.data.Data instance
    data_list = []
    for idx, (a_smarts, xyz_block) in tqdm.tqdm(enumerate(zip(rxn_smarts, xyz_blocks))):
        r, p = a_smarts.split(">>")
        data, feat_dict = generate_ts_data2(r, p, xyz_block, feat_dict=feat_dict)
        data_list.append(data)
        data.rxn_index = idx // 2
        data.augmented = False if idx % 2 == 0 else True

    # convert features to one-hot encoding
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.r_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.r_feat = torch.cat(feat_onehot, dim=-1)

        feat_onehot = []
        feats = data.p_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.p_feat = torch.cat(feat_onehot, dim=-1)

    train_index, valid_index, test_index = index_split(
        int(len(data_list) / 2),
        train=args.train,
        valid=args.valid,
        seed=args.seed
    )
    train_index = [i for i in train_index if i not in ban_index]
    valid_index = [i for i in valid_index if i not in ban_index]
    test_index = [i for i in test_index if i not in ban_index]

    train_data = [data_list[i] for i in train_index]
    valid_data = [data_list[i] for i in valid_index]
    test_data = [data_list[i] for i in test_index]
    index_dict = {
        "train_index": train_index,
        "valid_index": valid_index,
        "test_index": test_index,
    }

    # save the data, feat_dict, index_dict at the save_dir with pickle format. (.pkl)
    with open(os.path.join(args.save_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.save_dir, "valid_data.pkl"), "wb") as f:
        pickle.dump(valid_data, f)
    with open(os.path.join(args.save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    with open(os.path.join(args.save_dir, "feat_dict.pkl"), "wb") as f:
        pickle.dump(feat_dict, f)
    with open(os.path.join(args.save_dir, "index_dict.pkl"), "wb") as f:
        pickle.dump(index_dict, f)
