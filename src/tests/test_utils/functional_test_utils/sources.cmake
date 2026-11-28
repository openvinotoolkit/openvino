# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FUNCTIONAL_TEST_UTILS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/common_utils.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/crash_handler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/node_utils.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/ov_plugin_cache.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/skip_tests_config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/summary/api_summary.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/summary/environment.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/summary/op_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/summary/op_summary.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/summary/summary.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/functional_test_utils/test_model/test_model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/common_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/crash_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/precomp.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/skip_tests_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/summary/api_summary.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/summary/op_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/summary/op_summary.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/summary/summary.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/test_model/test_model.cpp
)
