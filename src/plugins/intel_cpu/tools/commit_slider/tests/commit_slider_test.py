# commit_slider_test.py
import sys

from unittest import TestCase
from tests import skip_commit_slider_devtest
sys.path.append('./')
from test_util import checkTestCase

class TryTesting(TestCase):
    @skip_commit_slider_devtest
    def testFbv(self):
        self.assertTrue(checkTestCase("FirstBadVersion"))

