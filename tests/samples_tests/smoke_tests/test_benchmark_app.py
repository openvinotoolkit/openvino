"""
 Copyright (C) 2018-2023 Intel Corporation
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
import os
import pytest
import subprocess
from common.samples_common_test_class import prepend



@pytest.mark.parametrize('counter', range(9999))
def test(counter, cache):
    try:
        subprocess.check_output(['benchmark_app', *prepend(cache, 'squeezenet1.1/FP32/squeezenet1.1.xml'), '-t', '1', '-api', 'sync'], stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=60.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(error.output)
        raise
