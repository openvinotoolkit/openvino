# Copyright (C) 2018-2020 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(vpu)

add_subdirectory(compile_tool)

# install

if(ENABLE_PYTHON)
    ie_cpack_add_component(python_tools REQUIRED DEPENDS core)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)
endif()
