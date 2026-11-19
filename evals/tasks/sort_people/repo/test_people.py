import unittest
from people import oldest_first


class T(unittest.TestCase):
    def test_order(self):
        self.assertEqual(oldest_first([('a', 30), ('b', 50), ('c', 10)]), [('b', 50), ('a', 30), ('c', 10)])
