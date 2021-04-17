"""
Gym environment with minesweeper implementation
"""

from typing import Iterable, Optional, Tuple
import gym
from gym.spaces import Box
from gym.spaces.multi_discrete import MultiDiscrete
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


class MineSweeper(gym.Env):
    """
    Class defining the game dynamics

    Note that, in this implementation, the first action will always
    happen in a cell with no bombs around it

    Args:
        nbombs: number of bombs in board
        shape:  (board height, board width)

    Examples:

        Create the environment and check that it's valid:

        ```python
        import gym
        env = gym.make("gym_minesweeper:minesweeper-v0")

        from stable_baselines3.common.env_checker import check_env
        check_env(env)
        ```
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nbombs: int = 10, shape: Tuple[int, int] = (8, 8)):
        assert min(shape) > 0 and nbombs > 0
        assert shape[0] * shape[1] > nbombs + 8

        self.observation_space = Box(low=-1, high=8, shape=shape, dtype=np.int8)

        self.action_space = MultiDiscrete(shape)

        self.nbombs = nbombs
        self.shape = shape

        self._bombs = None
        self._nnbombs = None
        self._layout = None  # type: Optional[np.ndarray]

        self._initialized = False
        self.done = False  # Set to False at reset()

        self.reset()

    @property
    def height(self) -> int:
        """Board height"""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Board width"""
        return self.shape[1]

    @property
    def ncells(self) -> int:
        """
        Number of cells in game (height * width)
        """
        return self.shape[0] * self.shape[1]

    def _step_one(self, action: Tuple[int, int]):
        """
        The first step will always hit a cell without bombs around it
        """

        a_y, a_x = action
        bombs_loc = [
            i
            for i in np.arange(self.ncells)
            if max(abs(i // self.width - a_y), abs(i % self.width - a_x)) > 1
        ]

        # Place all expected bombs
        self._bombs = np.zeros(self.shape, dtype=bool)
        idx = np.random.choice(bombs_loc, size=self.nbombs, replace=False)
        self._bombs[idx // self.width, idx % self.width] = True

        # Find number of bombs nearest neighbors of each cell
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self._nnbombs = convolve2d(self._bombs, kernel, "same")

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, int, int, dict]:
        """
        User perform an action

        Args:
            action: (row, column) clicked. 0 based

        Returns:
            state:  State after the action is executed
            reward: 1 if this action was the winning one, -1 if the losing one
            done:   True if the game is over
            info:   None
        """

        a_y, a_x = action
        v = self._layout[a_y, a_x]

        # First step? Decide where to locate bombs
        if not self._initialized:
            self._step_one(action)
            self._initialized = True

        # Game already over
        if self.done:
            return self._layout, 0, self.done, {}

        # Clicked on already visible cell
        if v != -1:
            return self._layout, -1, self.done, {}

        # Stepped on one bomb: game over
        if self._bombs[a_y, a_x]:
            self.done = True
            return self._layout, -self.shape[0] * self.shape[1], self.done, {}

        reward = 0

        # Open this cell
        self._layout[a_y, a_x] = self._nnbombs[a_y, a_x]
        reward += 1

        # Dummy code to open areas without neighboring bombs
        buffer = [action] if self._nnbombs[a_y, a_x] == 0 else []
        while buffer:
            buffer_new = []
            for row, col in buffer:
                for y in range(max(0, row - 1), min(self.shape[0], row + 2)):
                    for x in range(max(0, col - 1), min(self.shape[1], col + 2)):
                        if self._layout[y, x] >= 0:
                            # Already open cell
                            continue
                        if self._nnbombs[y, x] == 0:
                            buffer_new.append((y, x))
                        self._layout[y, x] = self._nnbombs[y, x]
                        reward += 1
            buffer = buffer_new

        # The game is won when the only non-visible cells are bombs
        if (self._layout == -1).sum() == self.nbombs:
            self.done = True

        return self._layout, reward, self.done, {}

    def reset(self):

        # Keep track of the state, as seen by the player
        self._layout = np.tile(np.int8(-1), self.shape)

        self._initialized = False  # To initialize after first step
        self.done = False
        return self._layout

    def render(  # pylint: disable=arguments-differ
        self, action: Optional[Iterable[int]] = None, mode: str = "human"
    ):
        """
        Simple rendering using matplotlib
        """
        assert mode in self.metadata["render.modes"]

        # Create figure with size proportional to the board size
        plt.figure(figsize=(0.25 * self.width, 0.25 * self.height))
        ax = plt.axes((0, 0, 1, 1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set(xlim=(0, self.width), ylim=(self.height, 0))

        # Draw numbers and gray out actions not available
        for nrow in range(self.height):
            for ncol in range(self.width):
                v = self._layout[nrow][ncol]
                if v == -1:
                    continue

                ax.add_patch(
                    patches.Rectangle(
                        xy=(ncol, nrow),
                        width=1,
                        height=1,
                        linewidth=0,
                        facecolor="#bbbbbb",
                    )
                )
                if v > 0:
                    ax.text(
                        x=ncol + 0.5,
                        y=nrow + 0.5,
                        s=v,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

        # Add lines between cells
        ax.hlines(np.arange(self.height), 0, self.width, "k", linewidth=1)
        ax.vlines(np.arange(self.width), 0, self.height, "k", linewidth=1)

        # Highlight last click, if given
        if action is not None:
            ax.add_patch(
                patches.Rectangle(
                    xy=(action[1], action[0]),
                    width=1,
                    height=1,
                    alpha=1,
                    facecolor="none",
                    linewidth=2,
                    edgecolor="#ff0000",
                )
            )

    def close(self):
        pass
