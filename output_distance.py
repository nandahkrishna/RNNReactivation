import json
import numpy as np
import pathlib
import torch

import config
import utils

args = config.load_analysis_config("output_distance")
args.path = pathlib.Path(args.path)

utils.set_random_seeds(args.seed)

expt_configs = [
    f"no-noise_unbiased_{args.seed}",
    f"no-noise_biased_{args.seed}",
    f"noisy_unbiased_{args.noise}_{args.seed}",
    f"noisy_biased_{args.noise}_{args.seed}",
]

distance = {"active": dict(), "quiescent": dict(), "quiescent_noisy": dict()}

for expt_config in expt_configs:
    try:
        model = torch.load(args.path / args.task / expt_config / f"model_{args.epoch}.pt")
    except:
        print(f"Skipped: {expt_config}.")
        continue

    model.device = args.device
    model.to(args.device)
    model.eval()

    task = model.task
    task.batch_size = args.batch_size

    test_data = task.get_test_batch()
    d = test_data["data"].shape[-1]
    quiescent_inputs = torch.zeros(task.batch_size, args.t_quiescent, d).to(args.device)

    _, h_active = model(test_data["data"], test_data["init_state"])
    model.sigma_rec *= np.sqrt(2)
    _, h_quiescent = model(quiescent_inputs, test_data["init_state"])

    active_xy = task.decode_outputs(h_active).cpu().detach().numpy()
    quiescent_xy = task.decode_outputs(h_quiescent).cpu().detach().numpy()

    distance["active"][expt_config] = utils.stepwise_distance(active_xy).tolist()
    distance["quiescent"][expt_config] = utils.stepwise_distance(quiescent_xy).tolist()

    if "no-noise" in expt_config:
        model.sigma_rec = np.sqrt(2 * args.noise)
        _, h_quiescent_noisy = model(quiescent_inputs, test_data["init_state"])
        quiescent_noisy_xy = task.decode_outputs(h_quiescent_noisy).cpu().detach().numpy()
        distance["quiescent_noisy"][expt_config] = utils.stepwise_distance(
            quiescent_noisy_xy
        ).tolist()

    print(f"Distance computed: {expt_config}.")

file_path = args.path / args.task / f"distance_{args.noise}_{args.seed}.json"
json.dump(distance, open(file_path, "w"), indent=4)
print(f"Results saved to {file_path}.")
