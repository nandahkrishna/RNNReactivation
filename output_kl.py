import json
import numpy as np
import pathlib
from scipy import stats
import torch

from models import rnn
import config
import utils

args = config.load_analysis_config("output_kde_kl")
args.path = pathlib.Path(args.path)

utils.set_random_seeds(args.seed)

expt_configs = [
    f"no-noise_unbiased_{args.seed}",
    f"no-noise_biased_{args.seed}",
    f"noisy_unbiased_{args.noise}_{args.seed}",
    f"noisy_biased_{args.noise}_{args.seed}",
]

kde = {"active": dict(), "quiescent": dict()}

for expt_config in expt_configs:
    try:
        model = torch.load(args.path / args.task / expt_config / f"model_{args.epoch}.pt")
    except:
        print(f"Skipped: {expt_config}.")
        continue

    model.set_device(args.device)
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

    active_kde = stats.gaussian_kde(active_xy.reshape(-1, d).T)
    print(f"KDE computed: {expt_config} (active).")
    quiescent_kde = stats.gaussian_kde(quiescent_xy.reshape(-1, d).T)
    print(f"KDE computed: {expt_config} (quiescent).")

    kde["active"][expt_config] = active_kde
    kde["quiescent"][expt_config] = quiescent_kde

random_net = rnn.RNN(
    task=model.task,
    n_in=model.n_in,
    n_rec=model.n_rec,
    n_out=model.n_out,
    n_init=model.n_init,
    sigma_in=model.sigma_in,
    sigma_rec=np.sqrt(args.noise),
    sigma_out=model.sigma_out,
    dt=model.dt,
    tau=model.tau,
    bias=model.bias,
    activation_fn=model.activation_fn,
    device=args.device,
)
random_net.set_device(args.device)
random_net.eval()

test_data = random_net.task.get_test_batch()
d = test_data["data"].shape[-1]
quiescent_inputs = torch.zeros(random_net.task.batch_size, args.t_quiescent, d).to(args.device)

random_net.sigma_rec *= np.sqrt(2)
_, h_quiescent = random_net(quiescent_inputs, test_data["init_state"])

quiescent_xy = random_net.task.decode_outputs(h_quiescent).cpu().detach().numpy()
quiescent_kde = stats.gaussian_kde(quiescent_xy.reshape(-1, d).T)
print(f"KDE computed: random (quiescent).")
kde["quiescent"]["random"] = quiescent_kde

kde["quiescent"]["uniform"] = stats.uniform(loc=-1.1, scale=2.2)
print("KDE computed: uniform (quiescent).")

kl = {key: dict() for key in kde["quiescent"].keys()}

for quiescent, quiescent_kde in kde["quiescent"].items():
    for active, active_kde in kde["active"].items():
        kl[quiescent][active] = utils.kl_divergence(active_kde, quiescent_kde)
print("KL divergences computed.")

file_path = args.path / args.task / f"kl_{args.noise}_{args.seed}.json"
json.dump(kl, open(file_path, "w"), indent=4)
print(f"Results saved to {file_path}.")
