# Copyright (C) 2018-2019 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
####################################

if (WIN32)
    # add_compile_options("/WX")
else()
    add_compile_options("-Werror")
endif()

if (ENABLE_OPENCV)
    add_subdirectory(calibration_tool/statistics_collector)
endif()

add_subdirectory(vpu)
