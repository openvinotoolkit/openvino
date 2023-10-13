# Copyright (c) 2019-2020 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Background:

    In the current implementation all internal e2e_oss imports are done assuming that working
    directory is `e2e_oss`, thus import like `from common.base_provider import BaseStepProvider`
    supposes import from `e2e.common.base_provider` module. This approach causes errors when
    importing e2e_oss functionality directly from other tests.

    The code below implements workaround for importing functionality from e2e_oss tests:
     - temporarily add e2e_oss path to sys.path to make it visible for importing
     - temporarily remove all instances of current working dir from sys.path to avoid possible
     importing from modules of the same name located in the current directory
     - restore original sys.path value after importing from e2e_oss

Usage:

    Add all imports from e2e_oss here and then use in the code like:
    `from mc_task_runner.common.e2e_wrapper import Environment`

"""

# pylint: disable=wrong-import-position,wrong-import-order,import-error,unused-import,no-name-in-module

from tests.utils.e2e.env_tools import Environment
from tests.utils.e2e.ir_provider.model_optimizer_runner import MORunner
from tests.utils.e2e.ref_collector.score_pytorch_onnx_runtime import PytorchPretrainedToONNXRunner,\
    PytorchTorchvisionToONNXRunner
