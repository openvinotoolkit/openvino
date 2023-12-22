# Copyright (C) 2018-2023 Intel Corporation
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

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == 'true', reason="Ticket - 113358")
def test_frontends():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'frontend_ngraph_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == 'true', reason="Ticket - 113358")
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


@pytest.mark.xfail(reason="Mismatched error messages due to namespace redesign.")
def test_main_error_log():
    setup_env()
    args = [sys.executable,
            os.path.join(os.path.dirname(__file__), 'main_test_error_log.py')]

    status = subprocess.run(args, env=os.environ, capture_output=True)
    test_log = status.stderr.decode("utf-8").replace("\r\n", "\n")

    # Check that log has exactly one warning from parse_args and
    # exactly one error message "FW ERROR"
    ref_log = "[ WARNING ]  warning\n[ FRAMEWORK ERROR ]  FW ERROR MESSAGE\n"

    assert test_log == ref_log


def test_mo_convert_logger():
    setup_env()
    args = [sys.executable,
            os.path.join(os.path.dirname(__file__), 'convert/logger_test_actual.py')]

    status = subprocess.run(args, env=os.environ, capture_output=True)
    test_log = status.stdout.decode("utf-8").replace("\r\n", "\n")

    assert "test message 1" in test_log
    assert "test message 2" in test_log
    assert "test message 3" in test_log

    assert test_log.count("[ SUCCESS ] Total execution time") == 2


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == 'true', reason="Ticket - 115084")
def test_rt_info():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'convert/meta_data_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ, capture_output=True)
    assert not status.returncode


def test_mo_extensions_test():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'extensions_test_actual.py'), '-s']

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


@pytest.mark.skipif(sys.version_info > (3, 10), reason="Ticket: 95904")
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


def test_convert_impl_tmp_irs_cleanup():
    setup_env()
    args = [sys.executable, '-m', 'pytest',
            os.path.join(os.path.dirname(__file__), 'utils', 'convert_impl_tmp_irs_cleanup_test_actual.py')]

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode
