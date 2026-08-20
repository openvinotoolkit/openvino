// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

void store_element(uint output_offset, uint output_size, uvec2 value) {
    uint output_byte_offset = output_offset * output_size;
    if (output_size == 1) {
        store_tail_u8(output_byte_offset, value.x);
    } else if (output_size == 2) {
        store_tail_u16(output_byte_offset, value.x);
    } else {
        uint output_word = output_byte_offset / 4;
        packed_output_data.values[output_word] = value.x;
        if (output_size == 8) {
            packed_output_data.values[output_word + 1] = value.y;
        }
    }
}
