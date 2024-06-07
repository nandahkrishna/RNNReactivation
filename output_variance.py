import json
import numpy as np
import pathlib
import torch

import config
import utils

args = config.load_analysis_config("output_variance")
args.path = pathlib.Path(args.path)

utils.set_random_seeds(args.seed)

expt_configs = [
    f"no-noise_unbiased_{args.seed}",
    f"no-noise_biased_{args.seed}",
    f"noisy_unbiased_{args.noise}_{args.seed}",
    f"noisy_biased_{args.noise}_{args.seed}",
]

variance = dict()

for expt_config in expt_configs:
    try:
        model = torch.load(args.path / args.task / expt_config / f"model_{args.epoch}.pt")
    except:
        print(f"Skipped: {expt_config}.")
        continue

    model.set_device(args.device)
    model.eval()
    model.sigma_rec *= np.sqrt(2)

    task = model.task
    task.batch_size = args.batch_size

    test_data = task.get_test_batch()
    d = test_data["data"].shape[-1]
    quiescent_inputs = torch.zeros(task.batch_size, args.t_quiescent, d).to(args.device)

    _, h_quiescent = model(quiescent_inputs, test_data["init_state"])
    quiescent_xy = task.decode_outputs(h_quiescent).cpu().detach().numpy()

    avg_var = utils.output_variance(quiescent_xy, shift=args.t_shift)
    variance[expt_config] = avg_var
    print(f"Variance computed: {expt_config}.")

file_path = args.path / args.task / f"variance_{args.noise}_{args.seed}.json"
json.dump(variance, open(file_path, "w"), indent=4)
print(f"Results saved to {file_path}.")
