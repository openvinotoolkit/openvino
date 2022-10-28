# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_classification
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES format_reader ie_samples_utils)

find_package(OpenCV QUIET COMPONENTS core)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV is disabled or not found, ${TARGET_NAME} will be built without OpenCV support. Set OpenCV_DIR")
else()
    target_compile_definitions(hello_classification PRIVATE USE_OPENCV)
    target_link_libraries(hello_classification PRIVATE opencv_core opencv_imgcodecs)
endif()

