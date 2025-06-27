// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_common_types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/visibility.hpp"

namespace ov::intel_cpu {

#if defined(OPENVINO_ARCH_ARM64)
#    define SNIPPETS_MAX_DATA_PTR_COUNT 23
#else
#    define SNIPPETS_MAX_DATA_PTR_COUNT 11
#endif

#define GET_OFF(field)           offsetof(ov::intel_cpu::jit_snippets_call_args, field)
#define GET_OFF_LOOP_ARGS(field) offsetof(ov::intel_cpu::jit_snippets_call_args::loop_args_t, field)

struct amx_tile_config_t {
    dnnl_dim_t M = 0;
    dnnl_dim_t K = 0;
    dnnl_dim_t N = 0;
};

struct jit_snippets_call_args {
    struct loop_args_t;

    jit_snippets_call_args() = default;
    ~jit_snippets_call_args();

    void register_loops(const std::vector<loop_args_t>& loops);
    void init_external_ptrs(size_t size);

    const void* src_ptrs[SNIPPETS_MAX_DATA_PTR_COUNT] = {};
    void* dst_ptrs[SNIPPETS_MAX_DATA_PTR_COUNT] = {};
    void* buffer_scratchpad_ptr = nullptr;

    // Note: Ideally loop_args must be private, since we manage this pointer manually.
    // However, standard-layout class definition (to use offset_of) requires the same access specifier
    // for all non-static data members. So we can keep them public or friend all control-flow emitters
    loop_args_t* loop_args = nullptr;
    amx_tile_config_t amx_tile_config;
    // Issue: 168073
    // TODO: decrease max array size
    size_t buffer_offsets[24] = {};
    const void** external_ptrs = nullptr;
};

struct jit_snippets_call_args::loop_args_t {
    loop_args_t() = default;
    loop_args_t(int64_t work_amount,
                const std::vector<int64_t>& ptr_increments,
                const std::vector<int64_t>& finalization_offsets);
    loop_args_t(const loop_args_t& other);
    ~loop_args_t();

    loop_args_t& operator=(loop_args_t other);
    friend void swap(loop_args_t& first, loop_args_t& second) noexcept;

    void init_pointers_and_copy_data(int64_t num_elements,
                                     const int64_t* ptr_increments,
                                     const int64_t* finalization_offsets);

    int64_t m_work_amount = 0;
    int64_t m_num_data_ptrs = 0;
    int64_t* m_ptr_increments = nullptr;
    int64_t* m_finalization_offsets = nullptr;
};

struct jit_snippets_compile_args {
    std::vector<std::vector<size_t>> data_offsets;
    std::vector<size_t> exec_domain;
};

}  // namespace ov::intel_cpu
