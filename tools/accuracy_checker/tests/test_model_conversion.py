"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import pytest

from accuracy_checker.launcher.model_conversion import (exec_mo_binary, find_dlsdk_ir, find_mo, prepare_args)
from tests.common import mock_filesystem


def test_mock_file_system():
    with mock_filesystem(['foo/bar', 'foo/baz/']) as prefix:
        assert (prefix / 'foo' / 'bar').is_file()
        assert (prefix / 'foo' / 'baz').is_dir()


def test_find_mo():
    with mock_filesystem(['deployment_tools/model_optimizer/mo.py']) as prefix:
        assert find_mo([prefix / 'deployment_tools' / 'model_optimizer'])


def test_find_mo_is_none_when_not_exist():
    with mock_filesystem(['deployment_tools/model_optimizer/mo.py']) as prefix:
        assert find_mo([prefix / 'deployment_tools']) is None


def test_find_mo_list_not_corrupted():
    with mock_filesystem(['deployment_tools/model_optimizer/mo.py']) as prefix:
        search_paths = [prefix]
        find_mo(search_paths)
        assert len(search_paths) == 1


def test_find_ir__in_root():
    with mock_filesystem(['model.xml', 'model.bin']) as root:
        model, weights = find_dlsdk_ir(root, 'model')
        assert model == root / 'model.xml'
        assert weights == root / 'model.bin'


def test_find_ir_raises_file_not_found_error_when_ir_not_found():
    with mock_filesystem(['foo/']) as root:
        with pytest.raises(FileNotFoundError):
            find_dlsdk_ir(root, 'model')


def test_prepare_args():
    args = prepare_args('foo', ['a', 'b'], {'bar': 123, 'x': 'baz'})
    assert args[0] == sys.executable
    assert args[1] == 'foo'
    assert '--a' in args
    assert '--b' in args
    assert '--bar' in args
    assert '--x' in args

    assert args[args.index('--bar') + 1] == '123'
    assert args[args.index('--x') + 1] == 'baz'


def test_exec_mo_binary(mocker):
    subprocess_run = mocker.patch('subprocess.run')
    mocker.patch('os.chdir')

    args = prepare_args('ModelOptimizer', value_options={'--foo': 'bar'})
    exec_mo_binary(args)

    subprocess_run.assert_called_once_with(args, check=False, timeout=None)
