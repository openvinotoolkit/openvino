################################################################################
# Copyright 2021 Intel Corporation
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
################################################################################

find_program(
    SPHINX_EXECUTABLE
    NAMES sphinx-build
    )

if (SPHINX_EXECUTABLE)
    execute_process(
        COMMAND "${SPHINX_EXECUTABLE}" --version
        OUTPUT_VARIABLE SPHINX_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _Sphinx_version_result
        )
    if (_Sphinx_version_result)
        message(WARNING "Unable to determine Sphinx version: ${_Sphinx_version_result}")
    endif()
    if(NOT TARGET Sphinx::sphinx)
        add_executable(Sphinx::sphinx IMPORTED GLOBAL)
        set_target_properties(Sphinx::sphinx PROPERTIES
            IMPORTED_LOCATION "${SPHINX_EXECUTABLE}"
        )
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Sphinx
    FOUND_VAR SPHINX_FOUND
    REQUIRED_VARS SPHINX_EXECUTABLE
    VERSION_VAR SPHINX_VERSION
)
