import unittest
from pal import is_palindrome


class T(unittest.TestCase):
    def test_mixed_case(self):
        self.assertTrue(is_palindrome('Never odd or even'))

    def test_not(self):
        self.assertFalse(is_palindrome('abc'))
