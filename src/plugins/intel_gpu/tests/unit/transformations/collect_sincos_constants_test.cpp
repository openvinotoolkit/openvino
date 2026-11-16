// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/increase_position_ids_precision.hpp"

namespace ov::test::intel_gpu {

// The ranges of FakeQuantize must stay foldable constants: they are not collected.
TEST(CollectSinCosConstantsForPrecisionPreservationTest, SkipsFakeQuantizeConstants) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 16});
    auto low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.f});
    auto high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.f});
    auto fake_quantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);
    auto coefficient = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16}, {6.2831855f});
    auto angle = std::make_shared<ov::op::v1::Multiply>(fake_quantize, coefficient);
    auto sin = std::make_shared<ov::op::v0::Sin>(angle);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});

    auto constants = std::make_shared<ov::intel_gpu::CollectSinCosConstantsForPrecisionPreservation::ConstantSet>();
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::CollectSinCosConstantsForPrecisionPreservation>(constants);
    manager.run_passes(model);

    EXPECT_EQ(constants->count(coefficient.get()), 1);
    EXPECT_EQ(constants->count(low.get()), 0);
    EXPECT_EQ(constants->count(high.get()), 0);
}

}  // namespace ov::test::intel_gpu
