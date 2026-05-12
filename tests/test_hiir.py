#!/usr/bin/env python3

from pyhiir.hiir import hiir
import unittest

class MainTest(unittest.TestCase):
    def test_create(self):
        h = hiir()
        self.assertIsNotNone(h)

if __name__=='__main__':
    unittest.main()
