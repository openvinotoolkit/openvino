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
    message(STATUS "NPU ${TARGET_NAME} tool is disabled due to missing dependencies: ${MISSING_DEPENDENCIES}")
    return()
endif()

#
# Define the target
#

ov_add_target(ADD_CPPLINT
              TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime
                      opencv_core
                      opencv_imgproc
                      opencv_imgcodecs
                      npu_tools_utils
                      gflags)

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

# TODO: fix warnings and remove this exception
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-missing-declarations)
endif()

#
# Install
#

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT}
        ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
            DESTINATION "tools/${TARGET_NAME}"
            COMPONENT ${NPU_INTERNAL_COMPONENT}
            ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})
endif()
