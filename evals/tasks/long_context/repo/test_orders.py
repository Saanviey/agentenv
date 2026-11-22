import unittest
from shop.orders import order_total


class T(unittest.TestCase):
    def test_bulk_order(self):
        self.assertEqual(order_total([50, 50], 10), 99.0)

    def test_small_order(self):
        self.assertEqual(order_total([10, 20], 2), 33.0)
