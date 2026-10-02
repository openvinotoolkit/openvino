# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_PROXY_PLUGIN_TESTS_SRC
    ${CMAKE_CURRENT_LIST_DIR}/batch_compliance_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/import_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/load_proxy_plugin_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/properties_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/proxy_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/proxy_tests.hpp
    ${CMAKE_CURRENT_LIST_DIR}/query_model_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/remote_context_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/remote_tensor_tests.cpp
)
