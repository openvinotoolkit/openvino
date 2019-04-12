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

from .dummy_launcher import DummyLauncher
from .launcher import Launcher, create_launcher, unsupported_launcher

try:
    from .caffe_launcher import CaffeLauncher
except ImportError as import_error:
    CaffeLauncher = unsupported_launcher(
        'caffe', "Caffe isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

try:
    from .dlsdk_launcher import DLSDKLauncher
except ImportError as import_error:
    DLSDKLauncher = unsupported_launcher(
        'dlsdk', "IE Python isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

__all__ = ['create_launcher', 'Launcher', 'CaffeLauncher', 'DLSDKLauncher', 'DummyLauncher']
