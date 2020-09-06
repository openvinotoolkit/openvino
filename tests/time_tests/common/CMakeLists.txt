# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "TimeTests")

file (GLOB SRC
        *.cpp
        ../ftti_pipeline/*.cpp)

file (GLOB HDR
        *.h
        ../ftti_pipeline/*.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

find_package(gflags REQUIRED)

target_link_libraries(${TARGET_NAME}
        gflags
        ${InferenceEngine_LIBRARIES}
        )
