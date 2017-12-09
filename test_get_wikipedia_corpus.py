from unittest import TestCase

from get_wikipedia_corpus import normalize_text


class GetWikipediaCorpus(TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text('′’ “some example”.'), '   " some example "  . ')
