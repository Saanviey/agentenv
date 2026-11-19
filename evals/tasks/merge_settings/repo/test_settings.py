import unittest
from settings import merge


class T(unittest.TestCase):
    def test_merge(self):
        d = {'a': 1, 'b': 2}
        self.assertEqual(merge(d, {'b': 3}), {'a': 1, 'b': 3})
        self.assertEqual(d, {'a': 1, 'b': 2})
