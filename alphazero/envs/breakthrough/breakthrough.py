from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from alphazero.Game import GameState

WIDTH, HEIGHT = 8, 8
# In principle we can reduce the action space by removing impossible moves, but
# having the empty spots makes it easier to implement symmetries and move
# validity checking. For each of the 3 move types (straight or the 2 diagonals)
# we mark the destination squares as valid.
ACTION_SIZE = 3 * WIDTH * HEIGHT

# State indices, constants for readability
PLAYER_W, PLAYER_B, CURR_PLAYER, CURR_PLY = 0, 1, 2, 3
EDGE_R = np.uint64(0x8080808080808080)
EDGE_L = np.uint64(0x0101010101010101)
BASE_W = np.uint64(0xFF00000000000000)
BASE_B = np.uint64(0x00000000000000FF)
HOME_W = np.uint64(0xFFFF000000000000)
HOME_B = np.uint64(0x000000000000FFFF)


@njit(parallel=True)
def explode3(a: np.uint64, b: np.uint64, c: np.uint64) -> NDArray[np.uint8]:
    """It happens to be a common operation here to make a 192-bit array from 3
    uint64s, and Numba supports neither unpackbits nor concatenate. We make do."""
    result = np.zeros((192,), dtype=np.uint8)
    for i in prange(64):
        result[i] = (a & (1 << i)) >> i
    for i in prange(64):
        result[i + 64] = (b & (1 << i)) >> i
    for i in prange(64):
        result[i + 128] = (c & (1 << i)) >> i
    return result


@njit
def nb_valid_moves(state: NDArray[np.uint64]) -> NDArray[np.uint8]:
    empty = ~(state[PLAYER_W] | state[PLAYER_B])
    if state[CURR_PLAYER] == PLAYER_W:
        straight = (state[PLAYER_W] >> 8) & empty
        diag_right = ((state[PLAYER_W] & ~EDGE_R) >> 7) & ~state[PLAYER_W]
        diag_left = ((state[PLAYER_W] & ~EDGE_L) >> 9) & ~state[PLAYER_W]
        return explode3(straight, diag_right, diag_left)

    else:
        straight = (state[PLAYER_B] << 8) & empty
        diag_right = ((state[PLAYER_B] & ~EDGE_R) << 9) & ~state[PLAYER_B]
        diag_left = ((state[PLAYER_B] & ~EDGE_L) << 7) & ~state[PLAYER_B]
        # If it's B to play we put the left moves first, which allows symmetry
        # to work as expected
        return explode3(straight, diag_left, diag_right)


@njit
def nb_play_action(state: NDArray[np.uint64], action: np.uint8):
    if state[CURR_PLAYER] == PLAYER_W:
        if action < 64:
            end_mask = np.uint64(1) << action
            start_mask = end_mask << 8
        elif action < 128:
            end_mask = np.uint64(1) << (action - 64)
            start_mask = end_mask << 7
        else:
            end_mask = np.uint64(1) << (action - 128)
            start_mask = end_mask << 9
        state[PLAYER_W] = (state[PLAYER_W] | end_mask) & ~start_mask
        state[PLAYER_B] = state[PLAYER_B] & ~end_mask
        state[CURR_PLAYER] = 1
        state[CURR_PLY] += 1
    else:
        if action < 64:
            end_mask = np.uint64(1) << action
            start_mask = end_mask << 8
        elif action < 128:
            end_mask = np.uint64(1) << (action - 64)
            start_mask = end_mask >> 7
        else:
            end_mask = np.uint64(1) << (action - 128)
            start_mask = end_mask >> 9
        state[PLAYER_W] = state[PLAYER_W] & ~end_mask
        state[PLAYER_B] = (state[PLAYER_B] | end_mask) & ~start_mask
        state[CURR_PLAYER] = 0
        state[CURR_PLY] += 1


@njit
def nb_win_state(state: NDArray[np.uint64]) -> NDArray[np.uint8]:
    return np.array(
        [
            np.uint8(state[PLAYER_W] & BASE_B > 0 or state[PLAYER_B] == 0),
            np.uint8(state[PLAYER_B] & BASE_W > 0 or state[PLAYER_W] == 0),
            0,
        ]
    )


@njit
def nb_observation(state: NDArray[np.uint64]) -> NDArray[np.float32]:
    return (
        explode3(
            state[PLAYER_W], state[PLAYER_B], state[CURR_PLAYER] * 0xFFFFFFFFFFFFFFFF
        )
        .reshape((3, 8, 8))
        .astype(np.float32)
    )

# No Numba for symmetry generation since it struggles a lot with all the
# reshapes, which should be pretty fast in base numpy. Any speed gain here is
# dwarfed during game generation.

def bin_fliph(i: np.uint64) -> np.uint64:
    """Given a u64 representing an 8x8 bitboard, flip all rows of that board"""
    i = int(i)
    i = ((i & 0xF0F0F0F0F0F0F0F0) >> 4) | ((i & 0x0F0F0F0F0F0F0F0F) << 4)
    i = ((i & 0xCCCCCCCCCCCCCCCC) >> 2) | ((i & 0x3333333333333333) << 2)
    i = ((i & 0xAAAAAAAAAAAAAAAA) >> 1) | ((i & 0x5555555555555555) << 1)
    return i


def bin_flipv(i: np.uint64) -> np.uint64:
    """Given a u64 representing an 8x8 bitboard, flip all columns of that board"""
    i = int(i)
    i = ((i & 0xFFFFFFFF00000000) >> 32) | ((i & 0x00000000FFFFFFFF) << 32)
    i = ((i & 0xFFFF0000FFFF0000) >> 16) | ((i & 0x0000FFFF0000FFFF) << 16)
    i = ((i & 0xFF00FF00FF00FF00) >> 8) | ((i & 0x00FF00FF00FF00FF) << 8)
    return np.uint64(i)


def nb_state_symmetries(state: NDArray[np.uint64]) -> NDArray[np.uint64]:
    return np.array(
        [
            [state[PLAYER_W], state[PLAYER_B], state[CURR_PLAYER], state[CURR_PLY]],
            [
                bin_fliph(state[PLAYER_W]),
                bin_fliph(state[PLAYER_B]),
                state[CURR_PLAYER],
                state[CURR_PLY],
            ],
            [
                bin_flipv(state[PLAYER_B]),
                bin_flipv(state[PLAYER_W]),
                1 - state[CURR_PLAYER],
                state[CURR_PLY],
            ],
            [
                bin_fliph(bin_flipv(state[PLAYER_B])),
                bin_fliph(bin_flipv(state[PLAYER_W])),
                1 - state[CURR_PLAYER],
                state[CURR_PLY],
            ],
        ],
        dtype=np.uint64,
    )


def nb_pi_symmetries(pi: NDArray[np.float32]) -> NDArray[np.float32]:
    pi_shape: NDArray[np.float32] = pi.reshape((3, 8, 8))
    return np.array(
        [
            pi,
            pi_shape[:, :, ::-1].flatten(),
            pi_shape[:, ::-1, :].flatten(),
            pi_shape[:, ::-1, ::-1].flatten(),
        ]
    )


class BreakthroughState(GameState):
    def __init__(self, state=None):
        if state is None:
            # State consists of white bitboard, black bitboard, current player
            # (0=w, 1=b), current turn. We keep the state in one nparray to
            # interact nicely with numba.
            self.state = np.array([HOME_W, HOME_B, PLAYER_W, 0], dtype=np.uint64)
        else:
            self.state = state

    def __str__(self) -> str:
        return f"Player:\t{self._player}\n{self._board}\n"

    def __eq__(self, other: "BreakthroughState") -> bool:
        """Compare the current game state to another"""
        return self.state == other.state

    def clone(self) -> "BreakthroughState":
        """Return a new clone of the game state, independent of the current one."""
        return BreakthroughState(np.copy(self.state))

    def action_size() -> int:
        """The size of the action space for the game"""
        return ACTION_SIZE

    def observation_size() -> Tuple[int, int, int]:
        """
        Returns:
            observation_size: the shape of observations of the current state,
                             must be in the form channels x width x height.
                             If only one plane is needed for observation, use 1 for channels.
        """
        return (3, WIDTH, HEIGHT)

    def valid_moves(self) -> np.ndarray:
        """Returns a numpy binary array containing zeros for invalid moves and ones for valids."""
        return nb_valid_moves(self.state)

    def num_players() -> int:
        """Returns the number of total players participating in the game."""
        return 2

    def max_turns() -> Optional[int]:
        """The maximum number of turns the game can last before a draw is declared."""
        return None

    def has_draw() -> bool:
        """Returns True if the game has a draw condition."""
        return False

    def player(self) -> int:
        return self.state[CURR_PLAYER]

    def turns(self) -> int:
        return self.state[CURR_PLY]

    def play_action(self, action: int) -> None:
        """Play the action in the current state given by argument action."""
        return nb_play_action(self.state, np.uint8(action))

    def win_state(self) -> np.ndarray:
        """
        Get the win state of the game, a numpy array of boolean values
        for each player indicating if they have won, plus one more
        boolean at the end to indicate a draw.
        """
        return nb_win_state(self.state)

    def observation(self) -> np.ndarray:
        """Get an observation from the game state in the form of a numpy array with the size of self.observation_size"""
        return nb_observation(self.state)

    def symmetries(self, pi) -> List[Tuple["BreakthroughState", np.ndarray]]:
        """
        Args:
            pi: the current policy for the given canonical state

        Returns:
            symmetries: list of state, pi pairs for symmetric samples of
                        the given state and pi (ex: mirror, rotation).
                        This is an optional method as symmetric samples
                        can be disabled for training.
        """
        states = nb_state_symmetries(self.state)
        pis = nb_pi_symmetries(pi)
        return list(zip(map(BreakthroughState, states), pis))
