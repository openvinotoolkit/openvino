#
# Copyright (C) 2018-2026 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME single-image-test)

if (NOT DEFINED PROJECT_NAME)
    if(WIN32)
        cmake_minimum_required(VERSION 3.16)
    else()
        cmake_minimum_required(VERSION 3.13)
    endif()
    project(single-image-test_standalone)
    include("cmake/standalone.cmake")
    return()
endif()

#
# Use OpenCV from Intel fork if it was fetched by parent CMakeLists.txt
#
set(OPENCV_FOUND_FROM_FORK FALSE)

if(NPU_OPENCV_FROM_FORK)
    message(STATUS "${TARGET_NAME}: Using OpenCV from Intel fork (already fetched)")
    set(OPENCV_FOUND_FROM_FORK TRUE)
    set(OPENCV_FORK_INCLUDE_DIRS ${NPU_OPENCV_FORK_INCLUDE_DIRS})
endif()

# Fall back to system OpenCV if fork fetch failed or was disabled
if(NOT OPENCV_FOUND_FROM_FORK)
    message(STATUS "${TARGET_NAME}: Looking for system OpenCV...")
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
endif()

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

ov_add_target(TYPE EXECUTABLE
              NAME ${TARGET_NAME}
              ROOT ${CMAKE_CURRENT_SOURCE_DIR}
              LINK_LIBRARIES
                  PRIVATE
                      openvino::runtime
                      opencv_imgcodecs
                      opencv_imgproc
                      opencv_core
                      npu_tools_utils
                      gflags)

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

# Use Intel fork's headers when building with FetchContent (must come before system OpenCV)
if(OPENCV_FOUND_FROM_FORK)
    target_include_directories(${TARGET_NAME} BEFORE PRIVATE ${OPENCV_FORK_INCLUDE_DIRS})
    if(NOT WIN32)
        target_link_options(${TARGET_NAME} PRIVATE -Wl,--no-as-needed)
    endif()
endif()

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
