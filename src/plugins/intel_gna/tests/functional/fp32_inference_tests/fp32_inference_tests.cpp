// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"
#include "test_models.hpp"

void run_test(std::shared_ptr<ov::Model> model,
              std::vector<std::vector<float>> input_data,
              std::vector<std::vector<float>> expected_result) {
    ov::Core core;
    ov::CompiledModel compiled_model =
        core.compile_model(model, "GNA", {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32)});
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    for (std::size_t i = 0; i < input_data.size(); i++) {
        ov::Tensor input_tensor = infer_request.get_input_tensor(i);
        auto data1 = input_tensor.data<float>();
        for (std::size_t j = 0; j < input_data[i].size(); j++) {
            data1[j] = input_data[i][j];
        }
    }

    infer_request.infer();

    for (std::size_t i = 0; i < expected_result.size(); i++) {
        ov::Tensor output_tensor = infer_request.get_output_tensor(i);
        auto out_data = output_tensor.data<float>();
        for (std::size_t j = 0; j < expected_result[i].size(); j++) {
            EXPECT_FLOAT_EQ(out_data[j], expected_result[i][j]);
        }
    }
}

TEST(Fp32InferenceTests, EltwiseAddModel) {
    std::vector<float> input_data(10, 1.0f);
    std::vector<float> expected_result(10, 3.5f);
    std::shared_ptr<ov::Model> model = eltwise_add_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, LSTMCellOnlyModel) {
    std::vector<float> input_data(96, 0.10f);
    std::vector<float> expected_result(32, 0.87488401f);
    std::shared_ptr<ov::Model> model = lstm_cell_only_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, LSTMCellOnlyModelUnaligned) {
    std::vector<float> input_data(30, 0.10f);
    std::vector<float> expected_result(10, 0.38111609f);
    std::shared_ptr<ov::Model> model = lstm_cell_only_model_unaligned();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, FCWithPaddingAfterSplitModel) {
    std::vector<float> input_data(20, 1.0f);
    std::vector<float> expected_result(10, 12.0f);
    std::shared_ptr<ov::Model> model = fc_with_padding_after_split_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, SliceModelWithAlignedOutputs) {
    std::vector<float> input_data(20, 1.0f);
    std::vector<float> expected_result(4, 18.0f);
    std::shared_ptr<ov::Model> model = slice_model_with_aligned_outputs();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, twoFCWithPaddingAfterSliceModel) {
    std::vector<float> input_data(20, 1.0f);
    std::vector<float> expected_result(8, 27.0f);
    std::shared_ptr<ov::Model> model = two_fc_with_padding_after_slice_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, ScaleShiftWithBroadcastSupported) {
    std::vector<float> input_data(40, 1.0);
    std::vector<float> expected_result = {2.0,  4.0,  6.0,  8.0,  10.0, 12.0, 14.0, 16.0, 2.0,  4.0,
                                          6.0,  8.0,  10.0, 12.0, 14.0, 16.0, 2.0,  4.0,  6.0,  8.0,
                                          10.0, 12.0, 14.0, 16.0, 2.0,  4.0,  6.0,  8.0,  10.0, 12.0,
                                          14.0, 16.0, 2.0,  4.0,  6.0,  8.0,  10.0, 12.0, 14.0, 16.0};
    std::shared_ptr<ov::Model> model = scaleshift_3d_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, InputSplitConcatPropagateForward) {
    std::vector<float> input_data(64, 1.0f);
    std::vector<float> expected_result(10, 129.f);
    std::shared_ptr<ov::Model> model = input_split_concat_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, InputSplitConcatUnaligned) {
    std::vector<float> input_data(20, 1.0f);
    std::vector<float> expected_result(10, 41.f);
    std::shared_ptr<ov::Model> model = input_split_concat_unaligned_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, PowerWithScaleFactorPropagateForward) {
    std::vector<float> input_data(10, 2.f);
    std::vector<float> expected_result(12, 21.f);
    std::shared_ptr<ov::Model> model = power_with_scale_factor_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, ReshapeConvolutionLessThan48Filters) {
    std::vector<float> input_data(800, 1.f);
    std::vector<float> expected_result(1600, 8.f);
    std::shared_ptr<ov::Model> model = reshape_convolution_less_than_48_filters();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{expected_result});
}

TEST(Fp32InferenceTests, multiple_inputs_correct_results) {
    std::vector<float> input_data(10, 1.0f);
    std::vector<float> input2_data(10, 2.0f);
    std::vector<float> result(10, 32.0f);
    std::shared_ptr<ov::Model> model = two_inputs_to_affine_model();
    run_test(model, std::vector<std::vector<float>>{input_data, input2_data}, std::vector<std::vector<float>>{result});
}

TEST(Fp32InferenceTests, TwoOutputsPropagateForward) {
    std::vector<float> input_data(10, 1.0f);
    std::vector<float> result1(20, 11.0f);
    std::vector<float> result2(10, 11.0f);
    std::shared_ptr<ov::Model> model = two_outputs_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{result1, result2});
}

TEST(Fp32InferenceTests, TwoOutputsDiffPrecisionPropagateForward) {
    std::vector<float> input_data(10, 1);
    std::vector<float> result1(20, 11.f);
    std::vector<float> result2(10, 11.f);
    std::shared_ptr<ov::Model> model = two_outputs_relu_model();
    run_test(model, std::vector<std::vector<float>>{input_data}, std::vector<std::vector<float>>{result1, result2});
}