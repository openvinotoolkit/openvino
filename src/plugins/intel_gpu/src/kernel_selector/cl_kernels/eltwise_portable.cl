// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/batch_headers/bf16_utils.cl"
#include "include/batch_headers/fetch_data.cl"

#if ELTWISE_PORTABLE_DENSE

// Non-scalar tensors share dense physical indexing; a scalar uses its own offset.
#    define GET_INDEX(prefix, num, idx_order) (CAT(CAT(prefix, num), _OFFSET) + (CAT(CAT(prefix, num), _LENGTH) == 1 ? 0 : element))

KERNEL(eltwise_portable)(INPUTS_DECLS __global OUTPUT_TYPE* output)
{
    // Spread groups over two axes without changing contiguous memory traversal.
    const uint row_begin = get_global_id(1) * ELTWISE_ROW_SIZE;
    const uint column = get_global_id(0);
#if ELTWISE_HAS_TAIL
    // Check before adding the column so a rounded tail cannot wrap the index.
    if (column >= ELTWISE_ELEMENTS_COUNT - row_begin)
        return;
#endif
    const uint element = row_begin + column;

    ACCUMULATOR_TYPE res;
    DO_ELTWISE;

#if QUANTIZATION_TERM && !OUTPUT_IS_FP
    output[OUTPUT_OFFSET + element] = TO_OUTPUT_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
#else
    output[OUTPUT_OFFSET + element] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(res, ACTIVATION_PARAMS_TYPED));
#endif
}

#else
// Keep layouts, broadcasting, dynamic shapes and fused operations in one source.
#include "generic_eltwise_ref.cl"
#endif
