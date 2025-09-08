# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "MemCheckTests")

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE StressTestsCommon)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

