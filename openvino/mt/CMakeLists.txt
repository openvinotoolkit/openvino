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

set(TARGET_NAME mt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# TBB library is stored on ftp
include(parallel)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(${TARGET_NAME} STATIC ${SOURCES})

add_library(openvino::mt ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX)
    target_compile_options(${TARGET_NAME} PUBLIC -Wno-undef)
endif()

set_ie_threading_interface_for(${TARGET_NAME})

# Install
if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    ie_cpack_add_component(tbb REQUIRED)
    list(APPEND core_components tbb)

    install(DIRECTORY "${TBB}/include"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    install(DIRECTORY "${TBB}/lib"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    if(EXISTS "${TBB}/bin")
      install(DIRECTORY "${TBB}/bin"
              DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
              COMPONENT tbb)
    endif()
    install(FILES "${TBB}/LICENSE"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    install(FILES "${TBB}/cmake/TBBConfig.cmake"
                  "${TBB}/cmake/TBBConfigVersion.cmake"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb/cmake
            COMPONENT tbb)
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/parallel.cmake"
        DESTINATION ${IE_CPACK_IE_DIR}/share
        COMPONENT core)
