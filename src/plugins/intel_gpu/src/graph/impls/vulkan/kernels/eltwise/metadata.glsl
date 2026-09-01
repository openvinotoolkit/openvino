// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

const uint byte_value_mask = 0xff;
const uint half_value_mask = 0xffff;

uint runtime_element_count() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.element_count;
#else
    return metadata.values[metadata_element_count];
#endif
}

uint runtime_python_division() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.python_division;
#else
    return metadata.values[metadata_python_division];
#endif
}

uint runtime_infinity_detection() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.infinity_detection;
#else
    return metadata.values[metadata_infinity_detection];
#endif
}

uint runtime_input0_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input0_coefficient;
#else
    return metadata.values[metadata_input0_coefficient];
#endif
}

uint runtime_input1_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input1_coefficient;
#else
    return metadata.values[metadata_input1_coefficient];
#endif
}

uint runtime_dense_input0_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input0_offset;
#else
    return metadata.values[header_words + tensor_input0 * tensor_words + max_rank * 2];
#endif
}

uint runtime_dense_input1_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.input1_offset;
#else
    return metadata.values[header_words + tensor_input1 * tensor_words + max_rank * 2];
#endif
}

uint runtime_dense_output_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.output_offset;
#else
    return metadata.values[header_words + tensor_output * tensor_words + max_rank * 2];
#endif
}

#if ELTWISE_FUSED
uint runtime_fused_input_offset() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_input_offset;
#else
    return metadata.values[metadata_fused_input_offset];
#endif
}

uint runtime_fused_python_division() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_python_division;
#else
    return metadata.values[metadata_fused_python_division];
#endif
}

uint runtime_fused_input0_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_input0_coefficient;
#else
    return metadata.values[metadata_fused_input0_coefficient];
#endif
}

uint runtime_fused_input1_coefficient() {
#if ELTWISE_DENSE_PUSH_CONSTANTS
    return dense_metadata.fused_input1_coefficient;
#else
    return metadata.values[metadata_fused_input1_coefficient];
#endif
}
#endif

#if ELTWISE_FUSED_CHAIN
uint fused_chain_metadata(uint stage, uint field) {
    return metadata.values[fused_metadata_base + stage * fused_metadata_count + field];
}

uint runtime_fused_chain_length() {
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH > 0
    return uint(ELTWISE_FIXED_FUSED_CHAIN_LENGTH);
#else
    return metadata.values[metadata_fused_chain_chain_length];
#endif
}
#endif
