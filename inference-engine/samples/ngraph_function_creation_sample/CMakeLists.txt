# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ngraph_function_creation_sample")

ie_add_sample(NAME ngraph_function_creation_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/ngraph_function_creation_sample.hpp"
              DEPENDENCIES format_reader ie_samples_utils)

find_package(ngraph REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES})
