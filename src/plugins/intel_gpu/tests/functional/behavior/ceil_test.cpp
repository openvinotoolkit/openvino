// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

TEST(BehaviorTest, CeilF64Check) {
    auto core = ov::Core();
    bool has_gpu = false;
    for (const auto& device : core.get_available_devices()) {
        if (device.find("GPU") != std::string::npos) {
            has_gpu = true;
            break;
        }
    }
    if (!has_gpu) {
        GTEST_SKIP() << "GPU device not available";
    }

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{1});
    auto ceil_op = std::make_shared<ov::op::v0::Ceiling>(param);
    auto result = std::make_shared<ov::op::v0::Result>(ceil_op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    ov::CompiledModel compiled_model;
    try {
        compiled_model = core.compile_model(model, "GPU");
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Failed to compile model: " << e.what();
    }
    
    auto req = compiled_model.create_infer_request();

    double input_val = 0.8214735388755798;
    ov::Tensor input_tensor(ov::element::f64, {1}, &input_val);
    req.set_input_tensor(input_tensor);
    req.infer();

    auto output_tensor = req.get_output_tensor();
    if (output_tensor.get_element_type() != ov::element::f64) {
         // If output converted to f32, we check that too, but expectation is f64 due to I/O preservation
         // However, if the fix works by converting internally, output should effectively be correct.
    }
    
    // Read back as double. If it was converted to f32, the runtime handles cast or we cast.
    // But data<double> assumes the tensor IS double.
    double output_val;
    if (output_tensor.get_element_type() == ov::element::f64) {
        output_val = output_tensor.data<double>()[0];
    } else if (output_tensor.get_element_type() == ov::element::f32) {
        output_val = static_cast<double>(output_tensor.data<float>()[0]);
    } else {
        GTEST_FAIL() << "Unexpected output type: " << output_tensor.get_element_type();
    }

    std::cout << "Input: " << input_val << ", Output: " << output_val << std::endl;
    ASSERT_EQ(output_val, 1.0);
}
