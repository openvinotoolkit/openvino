// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "frontend/shared/include/utils.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset14.hpp"
#include "paddle_utils.hpp"

using namespace ov::frontend;

// The linspace fuzzy case cannot show that the arithmetic is really done in float64: the CPU
// plugin executes float64 as float32, so the observed values stay within the fuzzy tolerance
// even if the graph is built in float32 and only the output is cast to float64. Check the
// converted graph instead, i.e. that neither the output nor the arithmetic was downgraded.
TEST(PaddleLinspaceTest, float64IsComputedInFloat64) {
    FrontEndManager fem;
    const auto frontend = fem.load_by_framework(PADDLE_FE);
    ASSERT_NE(frontend, nullptr);

    const auto model_path = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
                                                               "linspace_float64/linspace_float64.pdmodel");
    const auto input_model = frontend->load(model_path);
    ASSERT_NE(input_model, nullptr);

    const auto model = frontend->convert(input_model);
    ASSERT_NE(model, nullptr);

    // the regression of the issue: the declared float64 output used to be returned as float32
    ASSERT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_results()[0]->get_element_type(), ov::element::f64)
        << "the output of linspace(dtype=float64) must not be downgraded to float32";

    // the index range drives the whole arithmetic, so its type shows whether the computation
    // happens in float64 or the result is merely cast before the output
    std::shared_ptr<ov::op::v4::Range> range;
    for (const auto& op : model->get_ops()) {
        if ((range = ov::as_type_ptr<ov::op::v4::Range>(op))) {
            break;
        }
    }
    ASSERT_NE(range, nullptr) << "linspace must be converted using a Range operation";
    EXPECT_EQ(range->get_output_element_type(0), ov::element::f64)
        << "the range of linspace must be generated in float64, not only the output cast to it";
}
