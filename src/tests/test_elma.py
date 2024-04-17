import unittest
from src.hivit.vit import Elma


class TestElma(unittest.TestCase):

    def test_isim(self):
        elma = Elma()
        assert elma.isim == "Elma"
