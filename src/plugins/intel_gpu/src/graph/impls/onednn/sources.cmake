# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(GPU_ONEDNN_IMPL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/concatenation_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/concatenation_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/convolution_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/convolution_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/deconvolution_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/deconvolution_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fully_connected_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fully_connected_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gated_mlp_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gated_mlp_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/grouped_matmul_helper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gru_seq_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gru_seq_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/lstm_seq_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lstm_seq_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_gemm_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_gemm_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pooling_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pooling_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/primitive_onednn_base.h
    ${CMAKE_CURRENT_LIST_DIR}/reduce_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reduce_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/reorder_onednn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reorder_onednn.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.hpp
)
