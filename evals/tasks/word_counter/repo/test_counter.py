import unittest
from counter import WordCounter


class T(unittest.TestCase):
    def test_counts(self):
        c = WordCounter()
        for w in ['a', 'b', 'A']:
            c.add(w)
        self.assertEqual(c.counts['a'], 2)
        self.assertEqual(c.most_common(), 'a')
