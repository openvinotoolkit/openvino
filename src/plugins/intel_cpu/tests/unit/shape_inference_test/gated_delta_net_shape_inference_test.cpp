// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils.hpp"

using namespace ov::intel_cpu;
using ov::op::v0::Parameter;

struct GatedDeltaNetTestParams {
    StaticShapeVector static_input_shapes;
    StaticShapeVector expected_output_shapes;
};

class GatedDeltaNetStaticShapeInferenceTest : public OpStaticShapeInferenceTest<ov::op::internal::GatedDeltaNet> {
protected:
    void SetUp() override {
        const auto dynamic_rank4 = ov::PartialShape::dynamic(4);
        const auto dynamic_rank3 = ov::PartialShape::dynamic(3);
        auto q = std::make_shared<Parameter>(ov::element::f32, dynamic_rank4);
        auto k = std::make_shared<Parameter>(ov::element::f32, dynamic_rank4);
        auto v = std::make_shared<Parameter>(ov::element::f32, dynamic_rank4);
        auto state = std::make_shared<Parameter>(ov::element::f32, dynamic_rank4);
        auto g = std::make_shared<Parameter>(ov::element::f32, dynamic_rank3);
        auto beta = std::make_shared<Parameter>(ov::element::f32, dynamic_rank3);
        op = make_op(q, k, v, state, g, beta);
    }
};

class GatedDeltaNetStaticTestSuite : public GatedDeltaNetStaticShapeInferenceTest,
                                     public ::testing::WithParamInterface<GatedDeltaNetTestParams> {};

TEST_P(GatedDeltaNetStaticTestSuite, gated_delta_net_static_shape_inference) {
    const auto& [static_input_shapes, expected_output_shapes] = GetParam();

    auto shape_infer = make_shape_inference(op);
    EXPECT_EQ(shape_infer->get_port_mask(), EMPTY_PORT_MASK);

    const auto input_shape_refs = make_static_shape_refs(static_input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor());

    ASSERT_EQ(output_shapes.size(), expected_output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        EXPECT_EQ(output_shapes[i], expected_output_shapes[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(GatedDeltaNetStaticShapeInferenceTests,
                         GatedDeltaNetStaticTestSuite,
                         ::testing::Values(GatedDeltaNetTestParams{StaticShapeVector{StaticShape{1, 300, 4, 64},
                                                                                     StaticShape{1, 300, 4, 64},
                                                                                     StaticShape{1, 300, 4, 64},
                                                                                     StaticShape{1, 4, 64, 64},
                                                                                     StaticShape{1, 300, 4},
                                                                                     StaticShape{1, 300, 4}},
                                                                   StaticShapeVector{StaticShape{1, 300, 4, 64},
                                                                                     StaticShape{1, 4, 64, 64}}},
                                           GatedDeltaNetTestParams{StaticShapeVector{StaticShape{2, 128, 8, 32},
                                                                                     StaticShape{2, 128, 8, 32},
                                                                                     StaticShape{2, 128, 8, 32},
                                                                                     StaticShape{2, 8, 32, 32},
                                                                                     StaticShape{2, 128, 8},
                                                                                     StaticShape{2, 128, 8}},
                                                                   StaticShapeVector{StaticShape{2, 128, 8, 32},
                                                                                     StaticShape{2, 8, 32, 32}}}));
