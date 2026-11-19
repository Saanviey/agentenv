import unittest
from fb import fizzbuzz


class T(unittest.TestCase):
    def test_values(self):
        self.assertEqual([fizzbuzz(i) for i in (3, 5, 15, 7)], ['Fizz', 'Buzz', 'FizzBuzz', '7'])
