// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Parameter;

namespace {

std::shared_ptr<op::internal::GatedDeltaNet> make_gated_delta_net() {
    const auto dynamic = PartialShape::dynamic();
    auto q = std::make_shared<Parameter>(element::f32, dynamic);
    auto k = std::make_shared<Parameter>(element::f32, dynamic);
    auto v = std::make_shared<Parameter>(element::f32, dynamic);
    auto state = std::make_shared<Parameter>(element::f32, dynamic);
    auto g = std::make_shared<Parameter>(element::f32, dynamic);
    auto beta = std::make_shared<Parameter>(element::f32, dynamic);
    return std::make_shared<op::internal::GatedDeltaNet>(q, k, v, state, g, beta);
}

}  // namespace

TEST(StaticShapeInferenceTest, GatedDeltaNetBasic) {
    const auto op = make_gated_delta_net();

    // q, k, v: [1, 300, 4, 64]
    // state: [1, 4, 64, 64]
    // g, beta: [1, 300, 4]
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 300, 4, 64},
                                                    StaticShape{1, 300, 4, 64},
                                                    StaticShape{1, 300, 4, 64},
                                                    StaticShape{1, 4, 64, 64},
                                                    StaticShape{1, 300, 4},
                                                    StaticShape{1, 300, 4}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 2);
    EXPECT_EQ(static_output_shapes[0], StaticShape({1, 300, 4, 64}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({1, 4, 64, 64}));
}
