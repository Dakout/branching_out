# Branching Out: Prognostics-Based Replacement Policies for Series Systems

**Short Description:** 

This project implements various predictive maintenance policies for managing replacements in multi-component series systems. The input to the policies are component-specific remaining useful life (RUL) distributions. Here, these distributions are obtained by a provided multi-component RUL simulator.

---

## Installation

1. Prerequisites

    The most straightforward way to get the project up and running on your machine is to use the 'uv' manager. The code was developed on python 3.10; the package requirements can be found in the .toml and .lock files. 

    To install uv on your device, please visit [docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository
   ```bash
   git clone https://github.com/Dakout/branching_out.git
   cd branching_out

3. Set up the environment
    ```bash
    uv sync --locked

---

## Usage 

A tutorial notebook is provided, which shows how to:
1. quickly create a series system and evaluate a chosen policy's performance
2. create a series system with custom components, followed by a performance evaluation
3. evaluate a policy's performance for a limited given set of failure samples

There are many other customizations that one can implement, such as custom failure time distributions via the used ERADist package or new maintenance policies. 

I tried to document my code extensively and give the used functions expressive names. However, if things are still not clear or if you have any troubles in running the code, then please contact me via daniel.koutas@tum.de.

---

## Project Information
- **Authors:** Daniel Koutas, Daniel Straub
- **Organization:** ERA Group, TUM 
- **Publication:** Koutas, D., & Straub, D. (2026). Branching out: Prognostics-Based Replacement Policies for Series Systems. arXiv preprint [arXiv:2607.27899](https://arxiv.org/abs/2607.27899).

---

## Citation
If you find it useful for your research, please consider citing it:

**BibTeX Entry:**
```bibtex
@article{koutas2026branching,
  title={Branching out: Prognostics-Based Replacement Policies for Series Systems},
  author={Koutas, Daniel and Straub, Daniel},
  journal={arXiv preprint arXiv:2607.27899},
  year={2026}
}
```
(Feel free to also give the project a star on GitHub if you like it.)

---
