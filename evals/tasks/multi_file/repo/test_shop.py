import unittest
from shop import format_price


class T(unittest.TestCase):
    def test_price(self):
        self.assertEqual(format_price(1999), '$19.99')
