# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_UNIT_TEST_UTILS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/mocks/mock_dispatch_candidate/mock_dispatch_candidate.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/mock_engine/dllmain.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/mock_engine/mock_extensions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/mock_engine/mock_plugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/mock_engine/mock_plugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/openvino/runtime/mock_iasync_infer_request.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/openvino/runtime/mock_icompiled_model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/openvino/runtime/mock_icore.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/openvino/runtime/mock_iplugin.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/openvino/runtime/mock_isync_infer_request.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mocks/openvino/runtime/mock_ivariable_state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tests_data/istft_data.h
    ${CMAKE_CURRENT_LIST_DIR}/tests_data/roi_align_rotated_data.h
    ${CMAKE_CURRENT_LIST_DIR}/tests_data/scaled_dot_product_attention_data.h
    ${CMAKE_CURRENT_LIST_DIR}/tests_data/search_sorted_data.h
    ${CMAKE_CURRENT_LIST_DIR}/tests_data/sparse_fill_empty_rows_data.h
    ${CMAKE_CURRENT_LIST_DIR}/tests_data/stft_data.h
)
