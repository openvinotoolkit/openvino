#===============================================================================
# Copyright 2020-2021 Intel Corporation
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
#===============================================================================

# Controls testing options values and behavior
#===============================================================================

if(testing_cmake_included)
    return()
endif()
set(testing_cmake_included true)
include("cmake/options.cmake")

# Transfer string literal into a number to support nested inclusions easier
set(DNNL_TEST_SET_CI_NO_CORR "1")
set(DNNL_TEST_SET_CI "2")
set(DNNL_TEST_SET_NIGHTLY "3")

if(DNNL_TEST_SET STREQUAL "CI_NO_CORR")
    set(DNNL_TEST_SET ${DNNL_TEST_SET_CI_NO_CORR})
elseif(DNNL_TEST_SET STREQUAL "NIGHTLY")
    set(DNNL_TEST_SET ${DNNL_TEST_SET_NIGHTLY})
else()
    set(DNNL_TEST_SET ${DNNL_TEST_SET_CI})
endif()
