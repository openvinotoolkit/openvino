# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_CORE_TESTS_EXPERIMENTAL_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/reference/block_sparse_attention_eval.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reference/draft_block_mask_eval.cpp
    ${CMAKE_CURRENT_LIST_DIR}/type_prop/block_sparse_attention.cpp
    ${CMAKE_CURRENT_LIST_DIR}/type_prop/draft_block_mask.cpp
    ${CMAKE_CURRENT_LIST_DIR}/visitors/op/block_sparse_attention.cpp
    ${CMAKE_CURRENT_LIST_DIR}/visitors/op/draft_block_mask.cpp
)

set(OV_CORE_TESTS_EXPERIMENTAL_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/../reference/include
)
