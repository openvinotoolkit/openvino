# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_CORE_TESTS_REFERENCE_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/adaptive_rkv_diversity.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_cache_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/xattention.cpp
)

if(ENABLE_OV_EXPERIMENTAL)
    list(APPEND OV_CORE_TESTS_REFERENCE_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/block_sparse_attention_eval.cpp
        ${CMAKE_CURRENT_LIST_DIR}/draft_block_mask_eval.cpp)
endif()
