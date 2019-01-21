#===============================================================================
# Copyright 2018 Intel Corporation
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

if(profiling_cmake_included)
    return()
endif()
set(profiling_cmake_included true)

if("${VTUNEROOT}" STREQUAL "")
    message(STATUS "VTune profiling environment is unset")
else()
    set_ternary(JITPROFLIB MSVC "jitprofiling.lib" "libjitprofiling.a")
    list(APPEND EXTRA_LIBS "${VTUNEROOT}/lib64/${JITPROFLIB}")
    message(STATUS "VTune profiling environment is set")
endif()
