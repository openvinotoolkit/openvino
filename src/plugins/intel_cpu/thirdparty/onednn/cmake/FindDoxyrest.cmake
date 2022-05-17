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
    DOXYREST_EXECUTABLE
    NAMES doxyrest
    )

if (DOXYREST_EXECUTABLE)
    execute_process(
        COMMAND "${DOXYREST_EXECUTABLE}" --version
        OUTPUT_VARIABLE DOXYREST_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _Doxyrest_version_result
        )
    if (_Doxyrest_version_result)
        message(WARNING "Unable to determine doxyrest version: ${_Doxyrest_version_result}")
    endif()
    get_filename_component(DOXYREST_DIR ${DOXYREST_EXECUTABLE} ABSOLUTE)
    if(NOT TARGET Doxyrest::doxyrest)
        add_executable(Doxyrest::doxyrest IMPORTED GLOBAL)
        set_target_properties(Doxyrest::doxyrest PROPERTIES
            IMPORTED_LOCATION "${DOXYREST_EXECUTABLE}"
        )
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Doxyrest
    FOUND_VAR DOXYREST_FOUND
    REQUIRED_VARS DOXYREST_EXECUTABLE
    VERSION_VAR DOXYREST_VERSION
)
