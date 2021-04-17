
# Minesweeper gym environment

Simple Gym environment to play minesweeper.

## Installation

Let's create a simple conda environment to test this package:

```bash
conda create --name minesweeper-env -y -c conda-forge python=3.8 ipykernel
conda activate minesweeper-env
pip install -e .
```

Extra depencies used for the Reinforcement Learning notebooks can be installed
using (`--no-cache-dir` avoids out-of-memory problems installing torch):

```bash
pip --no-cache-dir install -r requirements.txt
```

## Useful links

* [Creating a Gym environment](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
