// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>

#include "cpu_types.h"
#include "nodes/executors/eltwise.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

#define MAX_ELTWISE_INPUTS   7
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_call_args_ptrs {
    const void* src_ptr[MAX_ELTWISE_INPUTS];
    void* dst_ptr;
    // ptr to array of post op inputs pointers (flat list)
    const void** post_op_data;

    // shape agnostic kernel
    size_t work_amount;
    const void* src_offsets[MAX_ELTWISE_INPUTS];
    const void* dst_offsets;
};

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    ov::element::Type src_prc[MAX_ELTWISE_INPUTS];
    ov::element::Type dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
    bool use_runtime_ptrs;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*){nullptr};

    void operator()(const jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);
        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(jit_eltwise_params jep) : jep_(std::move(jep)) {}
    virtual ~jit_uni_eltwise_kernel() = default;

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

class eltwise_precision_helper {
public:
    static ov::element::Type get_precision(const size_t inputs_number,
                                           const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                           const std::vector<EltwiseData>& eltwise_data,
                                           const std::vector<element::Type>& exec_precisions_priority);

private:
    static std::set<std::vector<element::Type>> get_supported_precisions(const Algorithm& algo);
};

}  // namespace ov::intel_cpu
