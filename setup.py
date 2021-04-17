"""
Install in edit mode using `pip install -e .`
"""

from setuptools import setup

setup(
    name="gym_minesweeper",
    description="Simple environment to play minesweeper",
    version="0.0.1",
    install_requires=["gym", "matplotlib", "numpy", "scipy"],
)
