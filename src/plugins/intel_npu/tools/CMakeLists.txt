#
# Copyright (C) 2018-2026 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
# Option to fetch OpenCV from Intel fork (shared by all NPU tools)
#
option(FETCH_OPENCV_FROM_FORK "Download OpenCV from Intel fork" ON)

set(NPU_OPENCV_FROM_FORK FALSE)

if(FETCH_OPENCV_FROM_FORK)
    message(STATUS "NPU Tools: Using OpenCV from Intel fork (FetchContent)")
    include(FetchContent)
    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/intel-innersource/frameworks.vpu.tools.opencv-intel-fork.git
        GIT_TAG        dee619b628e86d187bbe7a55b5fe6c7bfbb8b64d  # branch 4.x
    )
    # Build ALL required modules for all NPU tools
    # single-image-test: core, imgproc, imgcodecs
    # protopipe: + flann, features2d, calib3d, video, gapi
    # Note: video requires dnn for tracker_vit.cpp, so we include dnn as well
    set(BUILD_LIST "core;imgproc;imgcodecs;flann;features2d;calib3d;video;dnn;gapi" CACHE STRING "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
    # Disable 3rdparty libs that conflict with OpenVINO
    set(BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
    set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
    set(BUILD_ITT OFF CACHE BOOL "" FORCE)
    set(WITH_ITT OFF CACHE BOOL "" FORCE)
    set(ENABLE_PROFILING OFF CACHE BOOL "" FORCE)
    set(WITH_IPP OFF CACHE BOOL "" FORCE)
    set(BUILD_IPP_IW OFF CACHE BOOL "" FORCE)
    # Enable OpenVINO backend in G-API
    set(WITH_OPENVINO ON CACHE BOOL "" FORCE)
    set(OPENCV_GAPI_WITH_OPENVINO ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(opencv)
    message(STATUS "NPU Tools: OpenCV source dir: ${opencv_SOURCE_DIR}")

    # Check if all required targets were created successfully
    if(TARGET opencv_core AND TARGET opencv_imgproc AND TARGET opencv_imgcodecs)
        set(NPU_OPENCV_FROM_FORK TRUE CACHE INTERNAL "OpenCV was fetched from Intel fork")
        # Set include directories from the fork (must come before system OpenCV)
        set(NPU_OPENCV_FORK_INCLUDE_DIRS
            ${opencv_SOURCE_DIR}/modules/core/include
            ${opencv_SOURCE_DIR}/modules/imgproc/include
            ${opencv_SOURCE_DIR}/modules/imgcodecs/include
            ${opencv_SOURCE_DIR}/modules/flann/include
            ${opencv_SOURCE_DIR}/modules/features2d/include
            ${opencv_SOURCE_DIR}/modules/calib3d/include
            ${opencv_SOURCE_DIR}/modules/video/include
            ${opencv_SOURCE_DIR}/modules/dnn/include
            ${opencv_SOURCE_DIR}/modules/gapi/include
            ${opencv_BINARY_DIR}
            CACHE INTERNAL "OpenCV Intel fork include directories"
        )
        message(STATUS "NPU Tools: OpenCV from Intel fork: targets created successfully")
        message(STATUS "NPU Tools: OpenCV fork includes: ${NPU_OPENCV_FORK_INCLUDE_DIRS}")
        if(TARGET opencv_gapi)
            set(NPU_OPENCV_GAPI_FROM_FORK TRUE CACHE INTERNAL "OpenCV gapi was fetched from Intel fork")
            message(STATUS "NPU Tools: OpenCV gapi target also available")
        endif()
    else()
        message(WARNING "NPU Tools: OpenCV from Intel fork: failed to create targets, tools will fall back to system OpenCV")
    endif()
endif()

add_subdirectory(common)
add_subdirectory(compile_tool)
add_subdirectory(single-image-test)

if (ENABLE_INTEL_NPU_PROTOPIPE)
    add_subdirectory(protopipe)
endif()
