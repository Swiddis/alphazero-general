import unittest
import numpy as np

from alphazero.envs.breakthrough.breakthrough import BreakthroughState

class TestMoveGeneration(unittest.TestCase):
    def test_base_movecnt(self):
        self.assertEqual(BreakthroughState().valid_moves().sum(), 22)

    def test_base_movecnt_symmetric(self):
        mock_pi = np.zeros((192,), dtype=np.float32)
        for state, _ in BreakthroughState().symmetries(mock_pi):
            print('\nW: ' + hex(state.state[0]), 'B: ' + hex(state.state[1]), sep='\n')
            self.assertEqual(state.valid_moves().sum(), 22)
