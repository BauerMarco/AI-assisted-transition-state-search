import os
import pickle
import torch
from tqdm.auto import tqdm
import numpy as np

import re

from models import sampler
from torch_geometric.transforms import Compose
from models.epsnet import get_model
from utils.transforms import CountNodesPerGraph
from utils.misc import seed_all, get_logger
from torch_geometric.data import Batch
from sampling import preprocessing
from sampling import batching

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')

def smarts_to_symbols_ordered(smarts):
    """
    Extract element symbols ordered by atom mapping numbers (:1, :2, ...)
    from a SMARTS string.
    """

    reactant = smarts.split(">>")[0]

    # Match full bracket expressions: [O:1], [Cl:2], [N+:3], etc.
    tokens = re.findall(r'\[([^\]]+)\]', reactant)

    atom_map = {}

    for token in tokens:
        # Extract element symbol (e.g. O, Cl, N)
        elem_match = re.match(r'([A-Z][a-z]?)', token)
        if not elem_match:
            raise ValueError(f"Could not parse element from token: {token}")
        symbol = elem_match.group(1)

        # Extract mapping number after colon
        map_match = re.search(r':(\d+)', token)
        if not map_match:
            raise ValueError(f"No atom mapping found in token: {token}")
        idx = int(map_match.group(1))

        if idx in atom_map:
            raise ValueError(f"Duplicate atom map index found: {idx}")

        atom_map[idx] = symbol

    # Sort by mapping index
    max_idx = max(atom_map.keys())
    symbols = [atom_map[i] for i in range(1, max_idx + 1)]

    return symbols

def write_xyz(filename, symbols, coords, comment="Generated structure"):
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(comment + "\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


class sampling_args:
    def __init__(self):
        # Model
        self.ckpt = [
        "logs/trained_ckpt/ens0/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens1/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens2/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens3/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens4/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens5/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens6/checkpoints/best_ckpt.pt",
        "logs/trained_ckpt/ens7/checkpoints/best_ckpt.pt",
        ]
        # only use the first one for speed-up
        #self.ckpt = self.ckpts[0]
        self.device = "cpu"  #"cuda"
        self.batch_size = 100
        self.resume = None

        # IO parameters
        self.save_traj = False
        self.save_dir = "scratch/custom_run"  # REQUIRED in argparse

        # Test data parameters
        self.feat_dict = "./data/scratch/random_split_42/feat_dict.pkl"
        self.test_set = "data/scratch/random_split_42/test_data.pkl"  # REQUIRED
        self.start_idx = 0
        self.end_idx = 2
        self.repeat = 1

        # Guess TS parameters
        self.from_ts_guess = False
        self.denoise_from_time_t = None
        self.noise_from_time_t = None

        # Sampling parameters
        self.clip = 1000.0
        self.n_steps = 5000

        # Parameters for DDPM
        self.sampling_type = "ld"
        self.eta = 1.0
        self.step_lr = 1e-7

        # Misc
        self.seed = 2022

def sample(sampler_instance):
    args = sampler_instance

    # Logging
    # log_dir = args.save_dir
    log_dir = args.save_dir
    os.system(f"mkdir -p {log_dir}")
    logger = get_logger("test", log_dir)
    logger.info(args)

    # Load checkpoint
    logger.info("Loading model...")
    ckpts = [torch.load(x) for x in args.ckpt]
    models = []
    for ckpt, ckpt_path in zip(ckpts, args.ckpt):
        logger.info(f"load model from {ckpt_path}")
        model = get_model(ckpt["config"].model).to(args.device)
        model.load_state_dict(ckpt["model"])
        models.append(model)

    model = sampler.EnsembleSampler(models).to(args.device)
    seed_all(args.seed)

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = Compose([CountNodesPerGraph(), ])

    if ".txt" in args.test_set or ".pck" in args.test_set or ".pkl" in args.test_set:
        if not os.path.isfile(args.test_set):
            logger.info(f"!!!Test file {args.test_set} is not found!!!\n" * 3)
            exit()
        elif ".txt" in args.test_set:
            logger.info(f"Test file from {args.test_set}.\n Processing smarts...")
            smarts_list = open(args.test_set, "r").read().strip().split("\n")
            test_set = preprocessing(smarts_list, feat_dict_path=args.feat_dict)
        else:
            logger.info(f"Test file from {args.test_set}.\n Loading dataset...")
            test_set = pickle.load(open(args.test_set, "rb")) #"w"))
    else:
        logger.info(f"Test smarts : {args.test_set}.\n Processing smarts...")
        smarts_list = [args.test_set]
        test_set = preprocessing(smarts_list, feat_dict_path=args.feat_dict)

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx):
            continue
        test_set_selected.append(data)

    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)

    for i, batch in tqdm(enumerate(batching(test_set_selected, args.batch_size, repeat_num=args.repeat))):
        batch = Batch.from_data_list(batch).to(args.device)
        for _ in range(2):  # Maximum number of retry
            try:
                if args.from_ts_guess:
                    # print("Geometry Generation with Guess TS Support")
                    assert args.denoise_from_time_t is not None
                    if hasattr(batch, "ts_guess"):
                        init_guess = batch.ts_guess
                    else:
                        init_guess = batch.pos
                    start_t = (
                        args.noise_from_time_t
                        if args.noise_from_time_t is not None
                        else args.denoise_from_time_t
                    )
                    sqrt_a = model.alphas[start_t - 1].sqrt() if start_t != 0 else 1
                    init_guess = init_guess / sqrt_a
                    pos_init = init_guess.to(args.device)

                else:
                    pos_init = torch.randn(batch.num_nodes, 3).to(args.device)

                pos_gen, pos_gen_traj = model.dynamic_sampling(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=True,  # Done in transforms.
                    n_steps=args.n_steps,
                    step_lr=args.step_lr,
                    clip=args.clip,
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    noise_from_time_t=args.noise_from_time_t,
                    denoise_from_time_t=args.denoise_from_time_t,
                )
                alphas = model.alphas.detach()
                if args.denoise_from_time_t is not None:
                    alphas = alphas[args.denoise_from_time_t - args.n_steps: args.denoise_from_time_t]
                else:
                    alphas = alphas[model.num_timesteps - args.n_steps: model.num_timesteps]
                alphas = alphas.flip(0).view(-1, 1, 1)
                pos_gen_traj_ = torch.stack(pos_gen_traj) * alphas.sqrt().cpu()

                for j, data in enumerate(batch.to_data_list()):
                    mask = batch.batch == j
                    if args.save_traj:
                        data.pos_gen = pos_gen_traj_[:, mask]
                    else:
                        data.pos_gen = pos_gen[mask]

                    data = data.to("cpu")
                    results.append(data)
                    done_smiles.add(data.smiles)

                save_path = os.path.join(log_dir, "samples_not_all.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

                break  # No errors occured, break the retry loop
            except FloatingPointError:
                clip = 20
                logger.warning("Retrying with clipping thresh 20.")

    os.system(f"rm {save_path}")

    for i, res in enumerate(results):
        symbols = smarts_to_symbols_ordered(res.smiles)
        write_xyz(f"ts{i}.xyz", symbols, res.pos_gen)

