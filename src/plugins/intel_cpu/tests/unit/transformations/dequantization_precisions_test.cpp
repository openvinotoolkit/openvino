// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "nodes/fullyconnected.h"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

namespace {

class DequantizationPrecisionsTest : public ::testing::TestWithParam<ov::element::Type> {};

// A packed weight's Convert must survive ConstantFolding, or the payload is expanded to full precision.
TEST_P(DequantizationPrecisionsTest, PackedWeightConvertSurvivesConstantFolding) {
    const auto weight = std::make_shared<ov::op::v0::Constant>(GetParam(), ov::Shape{4, 8}, 1);
    const auto convert = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f16);
    const auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4, 1}, 1.0f);
    const auto multiply = std::make_shared<ov::op::v1::Multiply>(convert, scale);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MarkDequantization>(
        ov::intel_cpu::node::FullyConnected::getSupportedDequantizationPrecisions(true));
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.run_passes(model);

    const auto& ops = model->get_ordered_ops();
    const auto convert_it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return ov::as_type_ptr<ov::op::v0::Convert>(node) != nullptr;
    });
    ASSERT_NE(convert_it, ops.end());
    EXPECT_TRUE(ov::pass::constant_folding_is_disabled(*convert_it));
}

// u2 is already in the compressed-weight set: it is the control that shows the harness's dequantization chain
// (a {N,1} scale on a 2-D weight) satisfies the matcher, so an i2 failure is the set's and not the fixture's.
INSTANTIATE_TEST_SUITE_P(SubByte,
                         DequantizationPrecisionsTest,
                         ::testing::Values(ov::element::u2, ov::element::i2));

}  // namespace
