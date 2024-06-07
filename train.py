import numpy as np
from omegaconf import OmegaConf
import pathlib

from models import *
from tasks import *
import config
import trainer
import utils

args = config.load_train_config()

print("Experiment configuration:")
print(OmegaConf.to_yaml(args, resolve=True, sort_keys=True))

utils.set_random_seeds(args.seed)

pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)

if "spatial_navigation" in args.config:
    task = spatial_navigation.SpatialNavigation(
        box_width=args.task.box_width,
        box_height=args.task.box_height,
        border_region=args.task.border_region,
        border_slow_factor=args.task.border_slow_factor,
        init_pos=args.task.init_pos,
        biased=args.task.biased,
        drift_const=args.task.drift_const,
        anchor_point=np.array(args.task.anchor_point),
        dt=args.task.dt,
        mu=args.task.mu,
        sigma=args.task.sigma,
        b=args.task.b,
        use_place_cells=args.task.use_place_cells,
        place_cells_num=args.task.place_cells_num,
        place_cells_sigma=args.task.place_cells_sigma,
        place_cells_dog=args.task.place_cells_dog,
        place_cells_surround_scale=args.task.place_cells_surround_scale,
        sequence_length=args.task.sequence_length,
        batch_size=args.task.batch_size,
        device=args.device,
    )
elif "head_direction" in args.config:
    task = head_direction.HeadDirection(
        init_hd=args.task.init_hd,
        biased=args.task.biased,
        dt=args.task.dt,
        mu=args.task.mu,
        sigma=args.task.sigma,
        use_hd_cells=args.task.use_hd_cells,
        hd_cells_num=args.task.hd_cells_num,
        hd_cells_angular_spread=args.task.hd_cells_angular_spread,
        sequence_length=args.task.sequence_length,
        batch_size=args.task.batch_size,
        device=args.device,
    )
else:
    raise NotImplementedError("unknown task.")

train_data_generator = task.get_generator()
test_data = task.get_test_batch()

if args.rnn.model == "vanilla":
    model = rnn.RNN(
        task=task,
        n_in=args.rnn.n_in,
        n_rec=args.rnn.n_rec,
        n_out=args.rnn.n_out,
        n_init=args.rnn.n_init,
        sigma_in=np.sqrt(args.rnn.sigma2_in),
        sigma_rec=np.sqrt(args.rnn.sigma2_rec),
        sigma_out=np.sqrt(args.rnn.sigma2_out),
        dt=args.rnn.dt,
        tau=args.rnn.tau,
        bias=args.rnn.bias,
        activation_fn=args.rnn.activation_fn,
        device=args.device,
    )
elif args.rnn.model == "gru":
    model = gru.GRU(
        task=task,
        n_in=args.rnn.n_in,
        n_rec=args.rnn.n_rec,
        n_out=args.rnn.n_out,
        n_init=args.rnn.n_init,
        sigma_in=np.sqrt(args.rnn.sigma2_in),
        sigma_rec=np.sqrt(args.rnn.sigma2_rec),
        sigma_out=np.sqrt(args.rnn.sigma2_out),
        dt=args.rnn.dt,
        tau=args.rnn.tau,
        bias=args.rnn.bias,
        activation_fn=args.rnn.activation_fn,
        device=args.device,
    )
else:
    raise NotImplementedError("unknown model.")

model.to(args.device)
model_trainer = trainer.Trainer(
    model=model,
    train_data=train_data_generator,
    test_data=test_data,
    lr=args.trainer.lr,
    weight_decay=args.trainer.weight_decay,
    compute_all_metrics=args.trainer.compute_all_metrics,
    test_freq=args.trainer.test_freq,
    save_freq=args.trainer.save_freq,
    path=args.path,
    device=args.device,
)
model_trainer.train(n_epochs=args.trainer.n_epochs)
