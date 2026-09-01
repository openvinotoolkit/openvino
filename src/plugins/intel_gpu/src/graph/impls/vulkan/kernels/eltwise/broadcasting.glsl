// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if !ELTWISE_DENSE || ELTWISE_FUSED_BROADCAST
uint storage_element_offset_from_base(uint base, uint coordinates[max_rank], uint rank) {
    uint offset = metadata.values[base + max_rank * 2];
    for (uint axis = 0; axis < rank; ++axis) {
        uint dimension = metadata.values[base + axis];
        uint pitch = metadata.values[base + max_rank + axis];
        offset += (dimension == 1 ? 0 : coordinates[axis]) * pitch;
    }
    return offset;
}

#if !ELTWISE_DENSE
uint storage_element_offset(uint tensor_index, uint coordinates[max_rank], uint rank) {
    return storage_element_offset_from_base(header_words + tensor_index * tensor_words, coordinates, rank);
}
#endif

#if ELTWISE_FAST_BROADCAST || ELTWISE_FUSED_BROADCAST
uint fast_divide_u32(uint numerator, uint divisor, uint reciprocal) {
    if (divisor == 1) {
        return numerator;
    }
    uint product_high;
    uint product_low;
    umulExtended(numerator, reciprocal, product_high, product_low);
    uint remainder = numerator - product_high * divisor;
    return product_high + (remainder >= divisor ? 1u : 0u);
}

#endif

#if ELTWISE_FAST_BROADCAST
void fast_broadcast_offsets(uint linear_index, out uint input0_offset, out uint input1_offset, out uint output_offset) {
    uint input0_base = header_words + tensor_input0 * tensor_words;
    uint input1_base = header_words + tensor_input1 * tensor_words;
    uint output_base = header_words + tensor_output * tensor_words;
    input0_offset = metadata.values[input0_base + max_rank * 2];
    input1_offset = metadata.values[input1_base + max_rank * 2];
    output_offset = metadata.values[output_base + max_rank * 2];

    uint remainder = linear_index;
    for (int axis = int(selected_broadcast_rank) - 1; axis >= 0; --axis) {
        uint unsigned_axis = uint(axis);
        uint dimension = metadata.values[output_base + unsigned_axis];
        uint reciprocal = metadata.values[fast_divisor_metadata_base + unsigned_axis];
        uint quotient = fast_divide_u32(remainder, dimension, reciprocal);
        uint coordinate = remainder - quotient * dimension;
        uint axis_bit = 1u << unsigned_axis;
        if ((selected_broadcast_input0_axes & axis_bit) != 0) {
            input0_offset += coordinate * metadata.values[input0_base + max_rank + unsigned_axis];
        }
        if ((selected_broadcast_input1_axes & axis_bit) != 0) {
            input1_offset += coordinate * metadata.values[input1_base + max_rank + unsigned_axis];
        }
        if ((selected_broadcast_output_axes & axis_bit) != 0) {
            output_offset += coordinate * metadata.values[output_base + max_rank + unsigned_axis];
        }
        remainder = quotient;
    }
}
#endif
#endif

#if ELTWISE_FUSED_BROADCAST
uint fused_broadcast_input_offset(uint linear_index) {
    if ((selected_storage_flags & storage_fused_scalar_input_flag) != 0) {
        return metadata.values[fused_broadcast_metadata_base + max_rank * 2];
    }
    uint rank = metadata.values[metadata_rank];
    uint output_base = header_words + tensor_output * tensor_words;
    uint coordinates[max_rank];
    uint remainder = linear_index;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        uint unsigned_axis = uint(axis);
        uint dimension = metadata.values[output_base + unsigned_axis];
        uint reciprocal = metadata.values[fast_divisor_metadata_base + unsigned_axis];
        uint quotient = fast_divide_u32(remainder, dimension, reciprocal);
        coordinates[unsigned_axis] = remainder - quotient * dimension;
        remainder = quotient;
    }
    return storage_element_offset_from_base(fused_broadcast_metadata_base, coordinates, rank);
}
#endif
