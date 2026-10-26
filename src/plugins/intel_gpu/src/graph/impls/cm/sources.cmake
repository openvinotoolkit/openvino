# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(GPU_CM_IMPL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/include/cm_attention_common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/cm_pa_xe1.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/cm_pa_xe2.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/cm_qq_bias_mask.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/cm_sdpa_common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/estimate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/find_block.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/sort.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/xattn_subseq_meta.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pa_kv_reorder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pa_kv_reorder.hpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention.hpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention_gen.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention_gen.hpp
    ${CMAKE_CURRENT_LIST_DIR}/primitive_cm_base.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/kernel_generator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/kernel_generator.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/kernels_db.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/kernels_db.hpp
    ${CMAKE_CURRENT_LIST_DIR}/vl_sdpa_opt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vl_sdpa_opt.hpp
    ${CMAKE_CURRENT_LIST_DIR}/xetla_lstm_seq.cpp
    ${CMAKE_CURRENT_LIST_DIR}/xetla_lstm_seq.hpp
)

# .cm kernel files are not compiled directly; they are preprocessed/copied by
# the codegen custom commands below (tracked via DEPENDS for incremental builds).
set(GPU_CM_KERNEL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/cm_sdpa_vlen.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_kv_cache_reorder_ref.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_kv_cache_update_ref.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_multi_token.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_single_token.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_single_token_finalization.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_small_q.cm
    ${CMAKE_CURRENT_LIST_DIR}/pa_small_q_finalization.cm
    ${CMAKE_CURRENT_LIST_DIR}/xattn_find_block.cm
    ${CMAKE_CURRENT_LIST_DIR}/xattn_gemm_qk.cm
    ${CMAKE_CURRENT_LIST_DIR}/xattn_post_proc.cm
    ${CMAKE_CURRENT_LIST_DIR}/xetla_lstm_gemm.cm
    ${CMAKE_CURRENT_LIST_DIR}/xetla_lstm_loop.cm
)

set(GPU_CM_KERNEL_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/include/batch_headers/cm_xetla.h
    ${CMAKE_CURRENT_LIST_DIR}/include/xetla_lstm.h
)
