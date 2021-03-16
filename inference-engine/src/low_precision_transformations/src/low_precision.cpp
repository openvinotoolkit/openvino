// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"
#include <memory>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <low_precision/manager.hpp>

//#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"

// branch specific transformations
#include "low_precision/concat.hpp"
#include "low_precision/concat_multi_channels.hpp"

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/split.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"
#include "low_precision/split.hpp"

// cleanup transformations
#include "low_precision/convert.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::LowPrecision, "LowPrecision", 0);

ngraph::pass::low_precision::LowPrecision::LowPrecision(const LayerTransformation::Params params) : params(params){
    //
}

bool ngraph::pass::low_precision::LowPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    TransformationContext context(f);

    const std::vector<ngraph::element::Type> supportedTypes = { ngraph::element::i8, ngraph::element::u8 };
    ngraph::pass::Manager prerequisites;
    prerequisites.register_pass<PullReshapeThroughDequantization>(supportedTypes);
    prerequisites.register_pass<PullTransposeThroughDequantization>(supportedTypes);
    //prerequisites.register_pass<ngraph::pass::LinOpSequenceFusion>();
    prerequisites.run_passes(f);

    // step #1
    ngraph::pass::low_precision::Manager branchSpecificStep1(get_pass_config(), &context);
    branchSpecificStep1.register_pass<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, opset1::Concat>(params);
    branchSpecificStep1.run_passes(f);

    ngraph::pass::low_precision::Manager quantizationStep3(get_pass_config(), &context);
    quantizationStep3.register_pass<ngraph::pass::low_precision::AddTransformation, opset1::Add>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::AvgPoolTransformation, opset1::AvgPool>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::ClampTransformation, opset1::Clamp>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::ConvolutionTransformation, opset1::Convolution>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::DepthToSpaceTransformation, opset1::DepthToSpace>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::FakeQuantizeTransformation, opset1::FakeQuantize>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::GroupConvolutionTransformation, opset1::GroupConvolution>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::InterpolateTransformation, opset1::Interpolate>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::MatMulTransformation, opset1::MatMul>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::MaxPoolTransformation, opset1::MaxPool>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::MultiplyTransformation, opset1::Multiply>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::MVNTransformation, op::MVN>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::NormalizeL2Transformation, opset1::NormalizeL2>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::PReluTransformation, opset1::PRelu>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::ReluTransformation, opset1::Relu>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::ReshapeTransformation, opset1::Reshape>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::SqueezeTransformation, opset1::Squeeze>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::TransposeTransformation, opset1::Transpose>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::UnsqueezeTransformation, opset1::Unsqueeze>(params);
    quantizationStep3.register_pass<ngraph::pass::low_precision::InterpolateTransformation, ngraph::op::v4::Interpolate>(params);

    // step #2
    ngraph::pass::low_precision::Manager decompositionStep2(get_pass_config(), &context, &quantizationStep3);
    decompositionStep2.register_pass<ngraph::pass::low_precision::FakeQuantizeTransformation, opset1::FakeQuantize>(params);
    decompositionStep2.run_passes(f);

    // step #3
    quantizationStep3.run_passes(f);

    // step #4
    ngraph::pass::low_precision::Manager cleanupStep4(get_pass_config(), &context);
    cleanupStep4.register_pass<ngraph::pass::low_precision::FuseConvertTransformation, opset1::Multiply>(params);
    cleanupStep4.run_passes(f);

    // step #5
    ngraph::pass::low_precision::Manager cleanupStep5(get_pass_config(), &context);
    cleanupStep5.register_pass<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation, opset1::Subtract>(params);
    cleanupStep5.register_pass<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation, opset1::Multiply>(params);
    cleanupStep5.register_pass<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation, opset1::Multiply>(params);
    cleanupStep5.register_pass<ngraph::pass::low_precision::SubtractMultiplyToMultiplyAddTransformation, opset1::Multiply>(params);
    cleanupStep5.run_passes(f);

    return true;
}
