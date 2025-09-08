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
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME IntelSEAPI)

target_link_libraries(${TARGET_NAME} PRIVATE ittapi::ittnotify)

if(UNIX)
    target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})
elseif(WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE Dbghelp)
endif()

if(CMAKE_COMPILER_IS_GNUCC OR OV_COMPILER_IS_CLANG)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-undef -Wno-deprecated-declarations -Wno-multichar)
endif()

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        DESTINATION tests/sea_itt_lib COMPONENT itt_collector EXCLUDE_FROM_ALL)
