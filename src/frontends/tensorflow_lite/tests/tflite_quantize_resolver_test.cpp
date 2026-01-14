// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "tflite_ops/tflite_quantize.hpp"  // Internal header from src directory
#include "tflite_transformations/tflite_quantize_resolver.hpp"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite;

// Test that TFLQuantizeConvert pass doesn't optimize when quantize has multiple outputs
TEST(TFLQuantizeResolverTest, QuantizeConvertWithMultipleConsumers) {
    // Create a model: Parameter -> TFLQuantize -> [Convert -> Result, Result]
    // This simulates a case where the quantize output is consumed by both:
    // 1. A Convert node (converting to f32)
    // 2. A Result node directly (consuming the i8 output)
    auto param = std::make_shared<op::v0::Parameter>(element::f32, Shape{12});
    // Create quantization info
    std::vector<float> scale = {0.25f};
    std::vector<int64_t> zero_point = {16};
    auto quant_info = std::make_shared<QuantizationInfo>(scale, zero_point, 0);
    // TFLQuantize node (initially outputs i8)
    auto tfl_quantize = std::make_shared<TFLQuantize>(param, quant_info, element::i8);
    // Convert to f32
    auto convert = std::make_shared<op::v0::Convert>(tfl_quantize, element::f32);
    // Two results: one from convert (f32), one directly from quantize (i8)
    auto result1 = std::make_shared<op::v0::Result>(convert);
    auto result2 = std::make_shared<op::v0::Result>(tfl_quantize);
    auto model = std::make_shared<Model>(ResultVector{result1, result2}, ParameterVector{param});

    // Get initial type of result2 (should be i8)
    auto result2_type_before = model->get_results()[1]->get_element_type();
    EXPECT_EQ(result2_type_before, element::i8);
    // Run the TFLQuantizeConvert pass (this is the pass that was fixed)
    ov::pass::Manager manager;
    manager.register_pass<ov::frontend::tensorflow_lite::pass::TFLQuantizeConvert>();
    manager.run_passes(model);

    // Verify that result2 type remains i8
    auto result2_type_after = model->get_results()[1]->get_element_type();
    EXPECT_EQ(result2_type_after, element::i8) << "TFLQuantize type should remain i8";
}
