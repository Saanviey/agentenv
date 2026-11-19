import unittest
from mathx import safe_div


class T(unittest.TestCase):
    def test_div(self):
        self.assertEqual(safe_div(6, 3), 2)

    def test_zero(self):
        self.assertIsNone(safe_div(1, 0))
