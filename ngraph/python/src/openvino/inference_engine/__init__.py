# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from openvino.pyopenvino import IECore
from openvino.pyopenvino import IENetwork
from openvino.pyopenvino import ExecutableNetwork
from openvino.pyopenvino import Version
from openvino.pyopenvino import Parameter
from openvino.pyopenvino import InputInfoPtr
from openvino.pyopenvino import InputInfoCPtr
from openvino.pyopenvino import DataPtr
from openvino.pyopenvino import TensorDesc
from openvino.pyopenvino import get_version
from openvino.pyopenvino import StatusCode
from openvino.pyopenvino import Blob
from openvino.pyopenvino import PreProcessInfo
from openvino.pyopenvino import MeanVariant
from openvino.pyopenvino import ResizeAlgorithm
from openvino.pyopenvino import ColorFormat
from openvino.pyopenvino import PreProcessChannel

from openvino.inference_engine.ie_api import BlobPatch
# Patching for Blob class
Blob = BlobPatch
