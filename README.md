# Modeling Offline Reactivation or Replay in RNNs

:page_facing_up: [PDF](https://arxiv.org/pdf/2505.17003)
:globe_with_meridians: [arXiv](https://arxiv.org/abs/2505.17003)
:ballot_box_with_check: [OpenReview](https://openreview.net/forum?id=RVrINT6MT7)
:bar_chart: [Poster](https://iclr.cc/media/PosterPDFs/ICLR%202024/18641.png?t=1714898018.805508)
:movie_camera: [Presentation](https://iclr.cc/virtual/2024/poster/18641)

This repository contains code to reproduce experiments and figures from the paper ["Sufficient conditions for offline reactivation in recurrent neural networks"](https://arxiv.org/abs/2505.17003) published at [ICLR 2024](https://iclr.cc/Conferences/2024).

## Abstract

During periods of quiescence, such as sleep, neural activity in many brain circuits resembles that observed during periods of task engagement. However, the precise conditions under which task-optimized networks can autonomously reactivate the same network states responsible for online behavior is poorly understood. In this study, we develop a mathematical framework that outlines sufficient conditions for the emergence of neural reactivation in circuits that encode features of smoothly varying stimuli. We demonstrate mathematically that noisy recurrent networks optimized to track environmental state variables using change-based sensory information naturally develop denoising dynamics, which, in the absence of input, cause the network to revisit state configurations observed during periods of online activity. We validate our findings using numerical experiments on two canonical neuroscience tasks: spatial position estimation based on self-motion cues, and head direction estimation based on angular velocity cues. Overall, our work provides theoretical support for modeling offline reactivation as an emergent consequence of task optimization in noisy neural circuits.

**Keywords:** computational neuroscience, offline reactivation, replay, recurrent neural networks, path integration, noise

## Setup

Create a Python 3.10 virtual environment, and run the following command:

```zsh
pip install -r requirements.txt
```

## Experiments

To train a model for a specific configuration, you must run `train.py` with the right options:

```zsh
python train.py config=CONFIG_NAME seed=SEED [...]
```

For example, to train a noisy vanilla RNN on the unbiased spatial navigation task, you may run:

```zsh
python train.py config=spatial_navigation/noisy_unbiased seed=0
```

You may change the seed and any other hyperparameters or configuration variables either in the `.yml` files in [`configs/train`](/configs/train/), or pass them as command-line arguments. For example:

```zsh
python train.py config=spatial_navigation/noisy_unbiased seed=2 rnn.sigma2_rec=0.0003 trainer.n_epochs=1000 task.place_cells_num=256
```

After training models with different configurations and seeds, you may run the analysis scripts. Default arguments for these scripts are specified in the `.yml` files in [`configs/analysis`](/configs/analysis/). You may edit these configuration files or override default values using command-line arguments. For example, for models trained with seed 0 you may run:

```zsh
python output_kl.py seed=0
python output_variance.py seed=0
python output_distance.py seed=0
```

You may then use the Jupyter Notebooks in [`notebooks`](/notebooks/) to visualize the results.

### Reproducing Results

> [!NOTE]
> While it is possible to reproduce results from the paper overall, in practice the numbers and plots may not match _exactly_ due to differences in hardware, versions of CUDA, etc.

To reproduce experiments from the paper, run the following commands:

```zsh
for seed in {0..4}; do
    # Train vanilla RNNs on spatial navigation task
    python train.py config=spatial_navigation/noisy_unbiased seed=$seed
    python train.py config=spatial_navigation/noisy_biased seed=$seed
    python train.py config=spatial_navigation/no-noise_unbiased seed=$seed
    python train.py config=spatial_navigation/no-noise_biased seed=$seed

    # Analyze vanilla RNNs trained on spatial navigation task
    python output_kl.py seed=$seed
    python output_variance.py seed=$seed
    python output_distance.py seed=$seed

    # Train GRUs on spatial navigation task
    python train.py config=spatial_navigation/noisy_unbiased_gru seed=$seed
    python train.py config=spatial_navigation/noisy_biased_gru seed=$seed

    # Train vanilla RNNs on head direction task
    python train.py config=head_direction/noisy_unbiased seed=$seed
    python train.py config=head_direction/noisy_biased seed=$seed
done
```

Plots may then be generated using [`notebooks/SpatialNavigation.ipynb`](/notebooks/SpatialNavigation.ipynb) and [`notebooks/HeadDirection.ipynb`](/notebooks/SpatialNavigation.ipynb).

## License

This codebase is licensed under the BSD 3-Clause License (SPDX: `BSD-3-Clause`). Refer to [`LICENSE.md`](/LICENSE.md) for details.

## Citation

If this code was useful to you, please consider citing our work:

```bibtex
@inproceedings{krishna2024sufficient,
    title={Sufficient conditions for offline reactivation in recurrent neural networks},
    author={Nanda H Krishna and Colin Bredenberg and Daniel Levenstein and Blake Aaron Richards and Guillaume Lajoie},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=RVrINT6MT7}
}
```
