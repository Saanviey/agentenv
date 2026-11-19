import unittest
from listutil import last_n


class T(unittest.TestCase):
    def test_two(self):
        self.assertEqual(last_n([1, 2, 3, 4, 5], 2), [4, 5])

    def test_more_than_len(self):
        self.assertEqual(last_n([1, 2], 5), [1, 2])
