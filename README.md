# Offline reactivation in RNNs

:globe_with_meridians: [OpenReview](https://openreview.net/forum?id=RVrINT6MT7)
:page_facing_up: [PDF](https://openreview.net/pdf?id=RVrINT6MT7)
:bar_chart: [Poster](https://iclr.cc/media/PosterPDFs/ICLR%202024/18641.png?t=1714898018.805508)
:movie_camera: [Presentation](https://iclr.cc/virtual/2024/poster/18641)

This repository contains code to reproduce experiments and figures from the paper ["Sufficient conditions for offline reactivation in recurrent neural networks"](https://iclr.cc/virtual/2024/poster/18641) published at [ICLR 2024](https://iclr.cc/Conferences/2024).

## Abstract

During periods of quiescence, such as sleep, neural activity in many brain circuits resembles that observed during periods of task engagement. However, the precise conditions under which task-optimized networks can autonomously reactivate the same network states responsible for online behavior is poorly understood. In this study, we develop a mathematical framework that outlines sufficient conditions for the emergence of neural reactivation in circuits that encode features of smoothly varying stimuli. We demonstrate mathematically that noisy recurrent networks optimized to track environmental state variables using change-based sensory information naturally develop denoising dynamics, which, in the absence of input, cause the network to revisit state configurations observed during periods of online activity. We validate our findings using numerical experiments on two canonical neuroscience tasks: spatial position estimation based on self-motion cues, and head direction estimation based on angular velocity cues. Overall, our work provides theoretical support for modeling offline reactivation as an emergent consequence of task optimization in noisy neural circuits.

**Keywords:** computational neuroscience, offline reactivation, replay, recurrent neural networks, path integration, noise

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
