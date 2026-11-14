// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "frontend/shared/include/utils.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset6.hpp"
#include "paddle_utils.hpp"

using namespace ov::frontend;

// The alpha attribute of leaky_relu is always read as f32, so it has to be converted to
// the element type of the input before it is used as the PRelu slope. PRelu does not
// require its two inputs to have the same element type, so an f32 slope on f64 data is a
// valid graph, and the CPU plugin silently executes it in f32 - a value comparison
// (op_fuzzy) therefore cannot observe the missing conversion. This test checks the
// converted graph instead: it fails as soon as the conversion is dropped.
TEST(PaddleLeakyReluTest, slopeHasTheInputElementType) {
    FrontEndManager fem;
    const auto frontend = fem.load_by_framework(PADDLE_FE);
    ASSERT_NE(frontend, nullptr);

    const auto model_path = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
                                                               "leaky_relu_float64/leaky_relu_float64.pdmodel");
    const auto input_model = frontend->load(model_path);
    ASSERT_NE(input_model, nullptr);

    const auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);

    std::shared_ptr<ov::op::v0::PRelu> prelu;
    for (const auto& op : model->get_ops()) {
        if ((prelu = ov::as_type_ptr<ov::op::v0::PRelu>(op))) {
            break;
        }
    }
    ASSERT_NE(prelu, nullptr) << "leaky_relu must be converted to a PRelu operation";

    EXPECT_EQ(prelu->get_input_element_type(0), ov::element::f64)
        << "the data input of PRelu must keep the element type of the model input";
    EXPECT_EQ(prelu->get_input_element_type(1), ov::element::f64)
        << "the alpha attribute is read as f32 and must be converted to the element type of the data";
}
