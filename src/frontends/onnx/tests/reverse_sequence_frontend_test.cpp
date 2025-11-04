// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>  // <-- needed for std::ifstream

#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;
using namespace testing;

TEST(ONNXFrontendTests, ReverseSequenceZeroLen_Frontend) {
    // Use the pre-generated ONNX model placed under build/
    const std::string model_path = "temp_reverse_sequence.onnx";

    // Sanity: the test expects that file to exist
    std::ifstream f(model_path);
    ASSERT_TRUE(f.good())
        << "ONNX test model not found at: " << model_path
        << "\nPlease generate it (python tools/onnx/gen_reverse_sequence_onnx.py) and move it to build/";

    ov::frontend::FrontEndManager mgr;
    auto fe = mgr.load_by_framework("onnx");
    ASSERT_NE(fe, nullptr);

    auto fe_model = fe->load(model_path);
    ASSERT_NE(fe_model, nullptr);

    auto function = fe->convert(fe_model);
    ASSERT_NE(function, nullptr);

    ov::Core core;
    auto compiled = core.compile_model(function, "CPU");
    auto infer = compiled.create_infer_request();

    // Input: shape [4,4]
    ov::Shape shape{4, 4};
    ov::Tensor x(element::f32, shape);
    float vals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::memcpy(x.data<float>(), vals, sizeof(vals));

    // seq_lens: last batch zero-length
    ov::Tensor seq(element::i64, {4});
    int64_t seq_vals[] = {2, 1, 3, 0};
    std::memcpy(seq.data<int64_t>(), seq_vals, sizeof(seq_vals));

    infer.set_input_tensor(0, x);
    infer.set_input_tensor(1, seq);
    infer.infer();

    auto out = infer.get_output_tensor();
    const float* out_data = out.data<float>();
    auto shape_out = out.get_shape();

    ASSERT_EQ(shape_out, (ov::Shape{4, 4}));

    int zero_batch = 3;  // seq_len == 0
    for (size_t t = 0; t < shape_out[0]; ++t) {
        size_t idx = t * shape_out[1] + zero_batch;  // since batch_axis=1
        EXPECT_FLOAT_EQ(out_data[idx], 0.0f) << "Expected batch " << zero_batch << " at time " << t << " to be zero";
    }
}
