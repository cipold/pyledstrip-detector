#!/usr/bin/env python
# coding: utf-8

import unittest

from ledworld import LedWorld, Led


class TestParameters(unittest.TestCase):
    def test_single(self):
        world = LedWorld()
        world.add_led(Led(0, 0.0, 0.0))
        world.add_led(Led(1, 1.0, 1.0))
        self.assertEqual("""
[
  {
    "id": 0,
    "x": 0.0,
    "y": 0.0
  },
  {
    "id": 1,
    "x": 1.0,
    "y": 1.0
  }
]
        """.strip(), world.to_json())


if __name__ == '__main__':
    unittest.main()
