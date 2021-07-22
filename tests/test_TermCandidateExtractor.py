import unittest
import spacy
import spacy_canonicalizer.TermCandidateExtractor


class TestCandidateExtractor(unittest.TestCase):

    def __init__(self, arg, *args, **kwargs):
        super(TestCandidateExtractor, self).__init__(arg, *args, **kwargs)
