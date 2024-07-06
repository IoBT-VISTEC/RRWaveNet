import os
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

import models
from utils import resampling


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    model = getattr(models, args.model)(winsize=args.winsize,
                                        sampling_rate=args.sampling_rate,
                                        verbose=args.verbose)

    if args.show_summary:
        model.summary()

    need_co2 = args.model == "RespNet"

    # Load data
    x = np.load(f"data/{args.dataset}/X_{args.winsize}.npy", allow_pickle=True)
    y = np.load(f"data/{args.dataset}/RR_{args.winsize}.npy", allow_pickle=True)

    # Resampling

    x = resampling(x, args.winsize, args.sampling_rate)
    if need_co2:
        co2 = np.load(f"data/{args.dataset}/CO2_{args.winsize}.npy", allow_pickle=True)
        co2 = resampling(co2, args.winsize, args.sampling_rate)

    # Train
    folder = f"results/{args.dataset}/{args.model}_{args.winsize}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    if need_co2:
        model.train(x, co2, folder, rr=y)
    else:
        model.train(x, y, folder)


if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--model", type=str, default="RRWaveNet")
    args_parser.add_argument("--dataset", type=str, required=True)
    args_parser.add_argument("--winsize", type=int, required=True)
    args_parser.add_argument("--sampling_rate", type=int, default=50)
    args_parser.add_argument("--show_summary", action="store_true")
    args_parser.add_argument("--seed", type=int, default=69420)
    args_parser.add_argument("--verbose", type=int, default=0)

    main(args_parser.parse_args())
