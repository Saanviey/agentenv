import unittest
from text import reverse_words


class T(unittest.TestCase):
    def test_reverse(self):
        self.assertEqual(reverse_words('hello big world'), 'world big hello')
