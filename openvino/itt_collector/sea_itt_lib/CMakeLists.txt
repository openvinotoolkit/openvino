# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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

set(TARGET_NAME sea_itt_lib)

set(CMAKE_DEBUG_POSTFIX "")
set(CMAKE_RELEASE_POSTFIX "")

add_library(${TARGET_NAME} SHARED
    sea_itt_lib.cpp
    IttNotifyStdSrc.cpp
    IttNotifyStdSrc.h
    Recorder.cpp
    Recorder.h
    TraceEventFormat.h
    Utils.cpp
    Utils.h
)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME IntelSEAPI)

target_link_libraries(${TARGET_NAME} PRIVATE ittnotify)

if(UNIX)
    target_link_libraries(${TARGET_NAME} PRIVATE dl)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-undef -Wno-deprecated-declarations -Wno-multichar)
endif()
