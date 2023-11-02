# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
               ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Required zlib and cnpy dependencies

if(NOT TARGET ZLIB::ZLIB)
    if(EXISTS "${Samples_SOURCE_DIR}/thirdparty/zlib")
        # OpenVINO package puts thirdparty to samples dir
        add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/zlib"
                         "${Samples_BINARY_DIR}/thirdparty/zlib" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${Samples_SOURCE_DIR}/../../thirdparty/zlib")
        # Allow running samples CMakeLists.txt as stand alone from openvino sources
        add_subdirectory("${Samples_SOURCE_DIR}/../../thirdparty/zlib"
                         "${Samples_BINARY_DIR}/thirdparty/zlib" EXCLUDE_FROM_ALL)
    endif()
endif()

if(EXISTS "${Samples_SOURCE_DIR}/thirdparty/cnpy")
    # OpenVINO package puts thirdparty to samples dir
    add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/cnpy"
                     "${Samples_BINARY_DIR}/thirdparty/cnpy" EXCLUDE_FROM_ALL)
elseif(EXISTS "${Samples_SOURCE_DIR}/../../thirdparty/cnpy" AND NOT TARGET cnpy)
    # Allow running samples CMakeLists.txt as stand alone from openvino sources
    add_subdirectory("${Samples_SOURCE_DIR}/../../thirdparty/cnpy"
                     "${Samples_BINARY_DIR}/thirdparty/cnpy" EXCLUDE_FROM_ALL)
endif()

# add sample

ov_add_sample(NAME speech_sample
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES ${GFLAGS_TARGET} cnpy ie_samples_utils)
