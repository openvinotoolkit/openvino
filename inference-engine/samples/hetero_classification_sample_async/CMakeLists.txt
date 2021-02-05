# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hetero_classification_sample_async
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS hetero_classification_sample_async.h
              DEPENDENCIES format_reader)

find_package(ngraph REQUIRED)
target_link_libraries(hetero_classification_sample_async PRIVATE ${NGRAPH_LIBRARIES})
