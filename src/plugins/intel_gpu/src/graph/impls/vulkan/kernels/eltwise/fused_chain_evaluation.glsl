// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

uvec2 evaluate_fused_chain_stage(uvec2 result,
                                 uint stage,
                                 uint linear_index,
                                 uint output_type,
                                 uint mode,
                                 uint input_type,
                                 uint input_position) {
    uint input_offset = fused_chain_metadata(stage, fused_metadata_input_offset) + linear_index;
    if (is_float_type(output_type)) {
        float original = output_type == type_f32 ? uintBitsToFloat(result.x) : unpackHalf2x16(result.x).x;
        float external_value = load_float_fused_chain(stage, input_offset, input_type);
        float lhs = input_position == fused_input_lhs ? external_value : original;
        float rhs = input_position == fused_input_rhs ? external_value : original;
        return float_output_bits(apply_fused_float_runtime(lhs,
                                                           rhs,
                                                           mode,
                                                           fused_chain_metadata(stage, fused_metadata_input0_coefficient),
                                                           fused_chain_metadata(stage, fused_metadata_input1_coefficient)),
                                 output_type);
    }

    uvec2 external_value = load_integer_fused_chain(stage, input_offset, input_type);
    uvec2 lhs = input_position == fused_input_lhs ? external_value : result;
    uvec2 rhs = input_position == fused_input_rhs ? external_value : result;
    return apply_fused_integer_runtime(lhs,
                                       rhs,
                                       is_signed_type(output_type),
                                       mode,
                                       fused_chain_metadata(stage, fused_metadata_python_division) != 0);
}

uvec2 evaluate_element(uint linear_index, out uint output_offset) {
    uvec2 result = evaluate_base_element(linear_index, output_offset);
    uint output_type = selected_output_type;
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH > 0
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 1
    result = evaluate_fused_chain_stage(result,
                                        0,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_0,
                                        selected_fused_chain_input_type_0,
                                        selected_fused_chain_input_position_0);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 2
    result = evaluate_fused_chain_stage(result,
                                        1,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_1,
                                        selected_fused_chain_input_type_1,
                                        selected_fused_chain_input_position_1);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 3
    result = evaluate_fused_chain_stage(result,
                                        2,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_2,
                                        selected_fused_chain_input_type_2,
                                        selected_fused_chain_input_position_2);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 4
    result = evaluate_fused_chain_stage(result,
                                        3,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_3,
                                        selected_fused_chain_input_type_3,
                                        selected_fused_chain_input_position_3);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 5
    result = evaluate_fused_chain_stage(result,
                                        4,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_4,
                                        selected_fused_chain_input_type_4,
                                        selected_fused_chain_input_position_4);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 6
    result = evaluate_fused_chain_stage(result,
                                        5,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_5,
                                        selected_fused_chain_input_type_5,
                                        selected_fused_chain_input_position_5);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 7
    result = evaluate_fused_chain_stage(result,
                                        6,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_6,
                                        selected_fused_chain_input_type_6,
                                        selected_fused_chain_input_position_6);
#endif
#if ELTWISE_FIXED_FUSED_CHAIN_LENGTH >= 8
    result = evaluate_fused_chain_stage(result,
                                        7,
                                        linear_index,
                                        output_type,
                                        selected_fused_chain_mode_7,
                                        selected_fused_chain_input_type_7,
                                        selected_fused_chain_input_position_7);
#endif
#else
    uint chain_length = runtime_fused_chain_length();
    for (uint stage = 0; stage < chain_length; ++stage) {
        result = evaluate_fused_chain_stage(result,
                                            stage,
                                            linear_index,
                                            output_type,
                                            fused_chain_metadata(stage, fused_metadata_mode),
                                            fused_chain_metadata(stage, fused_metadata_input_type),
                                            fused_chain_metadata(stage, fused_metadata_input_position));
    }
#endif
    return result;
}
