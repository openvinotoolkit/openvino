# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_reshape_ssd
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/reshape_ssd_extension.hpp"
              DEPENDENCIES ie_samples_utils
              OPENCV_DEPENDENCIES core imgproc imgcodecs)

find_package(ngraph REQUIRED)
target_link_libraries(hello_reshape_ssd PRIVATE ${NGRAPH_LIBRARIES})
