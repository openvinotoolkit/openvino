// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#if ELTWISE_FUSED_CHAIN
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint values[];
} input1_data;

layout(set = 0, binding = 2) readonly buffer FusedInput0 {
    uint values[];
} fused_input0_data;
layout(set = 0, binding = 3) readonly buffer FusedInput1 {
    uint values[];
} fused_input1_data;
layout(set = 0, binding = 4) readonly buffer FusedInput2 {
    uint values[];
} fused_input2_data;
layout(set = 0, binding = 5) readonly buffer FusedInput3 {
    uint values[];
} fused_input3_data;
layout(set = 0, binding = 6) readonly buffer FusedInput4 {
    uint values[];
} fused_input4_data;
layout(set = 0, binding = 7) readonly buffer FusedInput5 {
    uint values[];
} fused_input5_data;
layout(set = 0, binding = 8) readonly buffer FusedInput6 {
    uint values[];
} fused_input6_data;
layout(set = 0, binding = 9) readonly buffer FusedInput7 {
    uint values[];
} fused_input7_data;

layout(set = 0, binding = 10) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint values[];
} output_data;
#elif ELTWISE_FUSED
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint values[];
} input1_data;

layout(set = 0, binding = 2) readonly buffer FusedInput {
    uint values[];
} fused_input_data;

layout(set = 0, binding = 3) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint values[];
} output_data;
#elif ELTWISE_UNARY
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint8_t values[];
} input0_data;

layout(set = 0, binding = 1) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint8_t values[];
} output_data;
#elif ELTWISE_SCALAR_CONSTANT
layout(set = 0, binding = 0) readonly buffer TensorInput {
    uint8_t values[];
} tensor_input_data;
#define input0_data tensor_input_data
#define input1_data tensor_input_data

layout(set = 0, binding = 1) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint8_t values[];
} output_data;
#elif ELTWISE_DENSE
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint values[];
} input1_data;

layout(set = 0, binding = 2) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint values[];
} output_data;
#else
layout(set = 0, binding = 0) readonly buffer Input0 {
    uint8_t values[];
} input0_data;

layout(set = 0, binding = 1) readonly buffer Input1 {
    uint8_t values[];
} input1_data;

layout(set = 0, binding = 2) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer Output {
    uint8_t values[];
} output_data;
#endif

#if ELTWISE_DENSE_PUSH_CONSTANTS
layout(push_constant) uniform DenseMetadata {
    uint element_count;
    uint input0_offset;
    uint input1_offset;
    uint output_offset;
    uint python_division;
    uint infinity_detection;
    uint input0_coefficient;
    uint input1_coefficient;
#if ELTWISE_FUSED
    uint fused_input_offset;
    uint fused_python_division;
    uint fused_input0_coefficient;
    uint fused_input1_coefficient;
#endif
} dense_metadata;
#else
#if ELTWISE_FUSED_CHAIN
layout(set = 0, binding = 11) readonly buffer Metadata {
#elif ELTWISE_FUSED
layout(set = 0, binding = 4) readonly buffer Metadata {
#elif ELTWISE_UNARY || ELTWISE_SCALAR_CONSTANT
layout(set = 0, binding = 2) readonly buffer Metadata {
#else
layout(set = 0, binding = 3) readonly buffer Metadata {
#endif
    uint values[];
} metadata;
#endif

#if ELTWISE_FUSED || ELTWISE_DENSE
#define packed_output_data output_data
#elif ELTWISE_UNARY || ELTWISE_SCALAR_CONSTANT
layout(set = 0, binding = 3) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer PackedOutput {
    uint values[];
} packed_output_data;
#else
layout(set = 0, binding = 4) writeonly ELTWISE_OUTPUT_ALIAS_QUALIFIER buffer PackedOutput {
    uint values[];
} packed_output_data;
#endif

#undef ELTWISE_OUTPUT_ALIAS_QUALIFIER
