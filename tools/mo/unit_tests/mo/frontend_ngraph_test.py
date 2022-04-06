# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from openvino.tools.mo.subprocess_main import setup_env, subprocess_main

import pytest


class TestNoInferenceEngine(unittest.TestCase):
    @patch('openvino.tools.mo.utils.find_ie_version.find_ie_version')
    def test_no_ie_ngraph(self, mock_find):
        mock_find.return_value = False
        with pytest.raises(SystemExit) as e, self.assertLogs(log.getLogger(), level="ERROR") as cm:
            subprocess_main()
        assert e.value.code == 1
        res = [i for i in cm.output if
               'Consider building the Inference Engine and nGraph Python APIs from sources' in i]
        assert res


def test_frontends():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'frontend_ngraph_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


def test_moc_extractor():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'moc_frontend/moc_extractor_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


def test_moc_preprocessing():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'back/moc_preprocessing_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


def test_main_test():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'main_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


def test_mo_extensions_test():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'extensions_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


def test_mo_fallback_test():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'utils/mo_fallback_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


def test_mo_model_analysis():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'utils/test_mo_model_analysis_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode
