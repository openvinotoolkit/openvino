#
# Copyright (C) 2022-2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME single-image-test)

if (NOT DEFINED PROJECT_NAME)
    cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
    project(single-image-test_standalone)
    include("cmake/standalone.cmake")
    return()
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

set(LINK_LIBRARIES
    openvino::runtime
    TBB::tbb
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    npu_tools_utils
    gflags
)

#
# check for missing dependencies
#

set(MISSING_DEPENDENCIES "")
foreach(LIB opencv_core opencv_imgproc opencv_imgcodecs)
    if(NOT TARGET ${LIB})
        list(APPEND MISSING_DEPENDENCIES ${LIB})
    endif()
endforeach()

if(NOT MISSING_DEPENDENCIES STREQUAL "")
    message(WARNING "${TARGET_NAME} tool is disabled due to missing dependencies: ${MISSING_DEPENDENCIES}")
    return()
endif()

#
# Define the target
#

ov_add_target(ADD_CPPLINT
              TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES PRIVATE ${LINK_LIBRARIES})

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

#
# Install
#

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT}
        EXCLUDE_FROM_ALL)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
            DESTINATION "tools/${TARGET_NAME}"
            COMPONENT ${NPU_INTERNAL_COMPONENT}
            EXCLUDE_FROM_ALL)
endif()
