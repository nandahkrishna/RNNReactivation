import numpy as np
from omegaconf import OmegaConf
import os
import pathlib
import torch

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("pi", lambda: np.pi)


def load_train_config():
    """Loads the experiment configuration for model training."""
    cli_config = OmegaConf.from_cli()

    if "base" in cli_config.config:
        raise ValueError("must not specify a base config file.")

    train_config_path = pathlib.Path("configs", "train")

    base_config = OmegaConf.load(train_config_path / "base.yml")

    task = cli_config.config.split(os.sep)[0]
    task_config = OmegaConf.load(train_config_path / task / "base.yml")

    expt_config = OmegaConf.load(train_config_path / f"{cli_config.config}.yml")
    expt_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    return OmegaConf.merge(base_config, task_config, expt_config, cli_config)


def load_analysis_config(analysis):
    """Loads the experiment configuration for post-training analyses."""
    cli_config = OmegaConf.from_cli()

    analysis_config_path = pathlib.Path("configs", "analysis")

    base_config = OmegaConf.load(analysis_config_path / "base.yml")

    expt_config = OmegaConf.load(analysis_config_path / f"{analysis}.yml")
    expt_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    return OmegaConf.merge(base_config, expt_config, cli_config)
