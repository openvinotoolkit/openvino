// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

void main() {
    uint invocation_index = gl_GlobalInvocationID.x;
    uint element_count = runtime_element_count();
    uint output_type = selected_output_type;
    uint output_size = scalar_size(output_type);
    bool pack_output = output_size < 4 && (selected_storage_flags & storage_packed_output_flag) != 0;
#if ELTWISE_DENSE || ELTWISE_BROADCAST_VECTOR || ELTWISE_SCALAR_CONSTANT
    uint elements_per_invocation = selected_elements_per_invocation;
#else
    uint elements_per_invocation = pack_output ? 4 / output_size : 1;
#endif
    uint first_element = invocation_index * elements_per_invocation;
    if (first_element >= element_count) {
        return;
    }

    uint first_output_offset;
    uvec2 first_value = evaluate_element(first_element, first_output_offset);
    if (!pack_output) {
        store_element(first_output_offset, output_size, first_value);
#if ELTWISE_DENSE || ELTWISE_BROADCAST_VECTOR || ELTWISE_SCALAR_CONSTANT
        for (uint vector_index = 1; vector_index < elements_per_invocation; ++vector_index) {
            uint element_index = first_element + vector_index;
            if (element_index >= element_count) {
                break;
            }
            uint output_offset;
            uvec2 value = evaluate_element(element_index, output_offset);
            store_element(output_offset, output_size, value);
        }
#endif
        return;
    }

    uint component_mask = output_size == 2 ? half_value_mask : byte_value_mask;
    uint packed_value = first_value.x & component_mask;
    uint processed_elements = 1;
    for (uint packed_index = 1; packed_index < elements_per_invocation; ++packed_index) {
        uint element_index = first_element + packed_index;
        if (element_index >= element_count) {
            break;
        }
        uint output_offset;
        uvec2 value = evaluate_element(element_index, output_offset);
        packed_value |= (value.x & component_mask) << (packed_index * output_size * 8);
        processed_elements += 1;
    }

    uint output_byte_offset = first_output_offset * output_size;
    if (processed_elements == elements_per_invocation) {
        packed_output_data.values[output_byte_offset / 4] = packed_value;
    } else if (output_size == 2) {
        store_tail_u16(output_byte_offset, packed_value);
    } else {
        for (uint byte_index = 0; byte_index < processed_elements; ++byte_index) {
            store_tail_u8(output_byte_offset + byte_index, packed_value >> (byte_index * 8));
        }
    }
}
