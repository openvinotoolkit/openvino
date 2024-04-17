// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_ARM64)
#define MAX_ELTWISE_INPUTS 14
#else
#define MAX_ELTWISE_INPUTS 7
#endif
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_call_args_ptrs {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst_ptr;
    //ptr to array of post op inputs pointers (flat list)
    const void** post_op_data;

    // shape agnostic kernel
    size_t work_amount;
    const void *src_offsets[MAX_ELTWISE_INPUTS];
    const void *dst_offsets;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov