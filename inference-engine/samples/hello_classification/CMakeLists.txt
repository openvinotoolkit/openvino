# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "hello_classification")

ie_add_sample(NAME hello_classification
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES ie_samples_utils
              OPENCV_DEPENDENCIES core imgcodecs)

find_package(ngraph REQUIRED)



#ie_add_sample(NAME ngraph_function_creation_sample
#        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
#        HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/ngraph_function_creation_sample.hpp"
#        DEPENDENCIES format_reader ie_samples_utils)

#find_package(ngraph REQUIRED)
find_package(OpenCV REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES} ${OpenCV_LIBRARIES})