// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// Regression test for the NCHW/NHWC layout mismatch in ConvertWeightCompressedConv1x1ToMatmul
// (PR #37133). It reproduces the machine-translation Transformer self-attention pattern that
// regressed on GPU: a weight-compressed 1x1 Convolution producing NCHW [N, Cout, 1, W] with the
// spatial size W > 1, whose output is consumed by a Reshape.
//
// ConvertWeightCompressedConv1x1ToMatmul rewrites the Convolution as a MatMul(transpose_b=true),
// whose output is channel-last NHWC [N, H, W, Cout] instead of the Convolution's channel-second
// NCHW [N, Cout, H, W]. Reshape never reorders elements, so feeding the NHWC MatMul output straight
// into a Reshape that was built for NCHW silently scrambles the channel/spatial ordering whenever
// H*W > 1. The reference is computed by the template plugin (which does not run this GPU-specific
// pass), so the scrambled GPU result no longer matches it and the test fails without the fix.
//
// The Convolution output must be static for the pass to fire (it bails out on dynamic output rank
// or dynamic C/H/W), so only the batch dimension is left dynamic here.
class Conv1x1WeightCompressedToMatmulReshapeConsumer : public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const size_t Cin = 128;
        const size_t Cout = 128;
        const size_t W = 55;  // spatial size > 1 - exposes the NHWC/NCHW mismatch

        // Batch is dynamic; the rest is static so the Convolution output stays static.
        InputShape data_shape{{-1, 1, static_cast<ov::Dimension::value_type>(W), static_cast<ov::Dimension::value_type>(Cin)},
                              {{1, 1, W, Cin}}};
        init_input_shapes({data_shape});

        const auto data_precision = ov::element::f16;
        auto data = std::make_shared<ov::op::v0::Parameter>(data_precision, inputDynamicShapes[0]);

        // [N, 1, W, Cin] -> Transpose[0,3,1,2] -> conv input NCHW [N, Cin, 1, W]
        auto in_transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto conv_input = std::make_shared<ov::op::v1::Transpose>(data, in_transpose_order);

        // Weight-compressed 1x1 convolution weights: i4 Constant -> Convert -> Multiply(scale).
        // The dequantization subgraph is kept from being constant-folded by the GPU pipeline's
        // MarkDequantization pass (which marks the Convert with disable_constant_folding), exactly as
        // for a real compressed model - so no manual marking is done here.
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i4,
                                                                      ov::Shape{Cout, Cin, 1, 1},
                                                                      ov::test::utils::InputGenerateData(-3, 6));
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, data_precision);

        auto scale_tensor = ov::test::utils::create_and_fill_tensor(data_precision,
                                                                    ov::Shape{Cout, 1, 1, 1},
                                                                    ov::test::utils::InputGenerateData(1, 4, 100));
        auto scale = std::make_shared<ov::op::v0::Constant>(scale_tensor);
        auto weights_dequant = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);

        auto conv = std::make_shared<ov::op::v1::Convolution>(conv_input,
                                                              weights_dequant,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);

        // conv output NCHW [N, Cout, 1, W] -> Reshape flattening features to [N, Cout*W]
        auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, -1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(conv, reshape_pattern, true);

        auto result = std::make_shared<ov::op::v0::Result>(reshape);
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{data},
                                               "Conv1x1CompressedReshapeConsumer");

        // f16 accumulation of a 128-wide reduction needs a slightly relaxed tolerance.
        abs_threshold = 0.5f;
        rel_threshold = 0.01f;
    }
};

TEST_F(Conv1x1WeightCompressedToMatmulReshapeConsumer, Inference) {
    run();
}
}  // namespace
