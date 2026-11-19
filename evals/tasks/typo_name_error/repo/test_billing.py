import unittest
from billing import total_price


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total_price([10, 20], 0.1), 33.0)
