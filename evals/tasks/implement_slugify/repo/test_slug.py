import unittest
from slug import slugify


class T(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify('Hello World!'), 'hello-world')

    def test_spaces(self):
        self.assertEqual(slugify('  Many   spaces here '), 'many-spaces-here')

    def test_hyphens(self):
        self.assertEqual(slugify('Already-Slugged'), 'already-slugged')
