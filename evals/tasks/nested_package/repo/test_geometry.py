import unittest
from geometry.shapes import circle, square


class T(unittest.TestCase):
    def test_circle(self):
        self.assertAlmostEqual(circle.area(3), 28.274, places=3)
        self.assertAlmostEqual(circle.circumference(3), 18.850, places=3)

    def test_square(self):
        self.assertEqual(square.area(3), 9)
