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
import pytest
import test


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="INTERPRETER",
        choices=["INTERPRETER", "CPU", "IE"],
        help="Select from available backends",
    )


def pytest_configure(config):
    backend_name = config.getvalue("backend")
    test.BACKEND_NAME = backend_name


def pytest_collection_modifyitems(config, items):
    backend_name = config.getvalue("backend")

    keywords = {
        "CPU": "skip_on_cpu",
        "INTERPRETER": "skip_on_interpreter",
        "IE": "skip_on_inference_engine",
    }

    skip_markers = {
        "CPU": pytest.mark.skip(reason="Skipping test on the CPU backend."),
        "INTERPRETER": pytest.mark.skip(reason="Skipping test on the INTERPRETER backend."),
        "IE": pytest.mark.skip(reason="Skipping test on the IE backend."),
    }

    for item in items:
        skip_this_backend = keywords[backend_name]
        if skip_this_backend in item.keywords:
            item.add_marker(skip_markers[backend_name])
