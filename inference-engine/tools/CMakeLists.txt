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

add_subdirectory(benchmark_app/cpp)

# install

if(ENABLE_PYTHON)
    ie_cpack_add_component(python_tools REQUIRED DEPENDS core)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_app/python
            DESTINATION deployment_tools/tools/benchmark_app
            COMPONENT python_tools)
    install(FILES
            COMPONENT python_tools)
endif()


ie_cpack_add_component(benchmark_app REQUIRED DEPENDS core)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_app/cpp
        DESTINATION deployment_tools/tools/benchmark_app
        COMPONENT benchmark_app
        USE_SOURCE_PERMISSIONS)
