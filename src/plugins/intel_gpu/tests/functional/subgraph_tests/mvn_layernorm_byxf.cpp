// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"

namespace {

// Regression: in f16 the convolutions + residual pin the MVN input to a non-planar byxf layout, where the GPU
// normalized a last-axis (axes=[-1]) LayerNorm over the wrong axis (in f32 it stays planar bfyx and is correct).
//
//                  ┌───────┐
//                  │ Param │
//                  └───┬───┘
//                  ┌───┴───┐
//          ┌───────┤ Conv1 │
//          │       └───┬───┘
//          │     ┌─────┴─────┐
//          │     │ Transpose │ (bfyx->byxf)
//          │     └─────┬─────┘
//          │       ┌───┴───┐
//          │       │  MVN  │ (axes=[-1])
//          │       └───┬───┘
//          │     ┌─────┴────┐
//          │     │ Multiply │ (gamma)
//          │     └─────┬────┘
//          │        ┌──┴──┐
//          │        │ Add │ (beta)
//          │        └──┬──┘
//          │     ┌─────┴─────┐
//          │     │ Transpose │ (byxf->bfyx)
//          │     └─────┬─────┘
//          │       ┌───┴───┐
//          │       │ Conv2 │
//          │       └───┬───┘
//          │        ┌──┴──┐
//          └────────┤ Add │ (residual)
//                   └──┬──┘
//               ┌──────┴─────┐
//               │   Result   │
//               └────────────┘
class MVNLayerNormByxf : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto type = ov::element::f16;

        // Channel count is kept below 16 so the GPU layout optimizer selects the byxf layout for the MVN input.
        const ov::Shape input_shape{1, 8, 64, 64};
        const size_t channels = input_shape[1];

        // The wrong-axis MVN corrupts the majority of the output elements (relative error ~0.1), while a correct f16
        // result stays well within these thresholds at the output magnitude produced by the bounded weights below.
        abs_threshold = 0.1f;
        rel_threshold = 0.05f;

        // Deterministic constant values in [-1, 1) keep activations bounded so the f16 reference comparison is clean.
        auto bounded_const_data = [](int32_t seed) {
            return ov::test::utils::InputGenerateData(-1, 2, 32, seed);
        };

        auto input = std::make_shared<ov::op::v0::Parameter>(type, input_shape);

        // First convolution: keeps the activation in a channels-last (byxf) GPU layout.
        auto weights1 = ov::test::utils::make_constant(type, ov::Shape{channels, channels, 3, 3}, bounded_const_data(1));
        auto conv1 = std::make_shared<ov::op::v1::Convolution>(input,
                                                               weights1,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::Strides{1, 1});

        // LayerNorm over the channel (f) axis, expressed as Transpose -> MVN(axes=[-1]) -> scale/shift -> Transpose.
        auto order_to_byxf = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        auto transpose_in = std::make_shared<ov::op::v1::Transpose>(conv1, order_to_byxf);

        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mvn = std::make_shared<ov::op::v6::MVN>(transpose_in,
                                                     axes,
                                                     true,
                                                     1e-6f,
                                                     ov::op::MVNEpsMode::INSIDE_SQRT);

        auto gamma = ov::test::utils::make_constant(type, ov::Shape{1, 1, 1, channels}, bounded_const_data(2));
        auto scale = std::make_shared<ov::op::v1::Multiply>(mvn, gamma);
        auto beta = ov::test::utils::make_constant(type, ov::Shape{1, 1, 1, channels}, bounded_const_data(3));
        auto shift = std::make_shared<ov::op::v1::Add>(scale, beta);

        auto order_to_bfyx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose_out = std::make_shared<ov::op::v1::Transpose>(shift, order_to_bfyx);

        // Second convolution + residual connection that pins the byxf layout on the LayerNorm block.
        auto weights2 = ov::test::utils::make_constant(type, ov::Shape{channels, channels, 3, 3}, bounded_const_data(4));
        auto conv2 = std::make_shared<ov::op::v1::Convolution>(transpose_out,
                                                               weights2,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::Strides{1, 1});
        auto residual = std::make_shared<ov::op::v1::Add>(conv2, conv1);

        auto result = std::make_shared<ov::op::v0::Result>(residual);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "MVNLayerNormByxf");
    }
};

TEST_F(MVNLayerNormByxf, smoke_GPU_MVNLayerNormByxf) {
    run();
}

}  // namespace
