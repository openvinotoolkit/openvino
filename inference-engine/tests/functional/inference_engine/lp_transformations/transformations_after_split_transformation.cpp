// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include <transformations/utils/utils.hpp>

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
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
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"

// cleanup transformations
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

#include "lpt_ngraph_functions/transformations_after_split_function.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"


namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

void getTransformerWithTransformationByName(
    SimpleLowPrecisionTransformer& transformer,
    const TestTransformationParams& params,
    const std::string name) {
    using namespace pass::low_precision;

    if (name == "AddTransformationWithoutConcat" || name == "AddTransformationWithConcat") {
        transformer.add<AddTransformation, ngraph::opset1::Add>(params);
        return;
    }
    if (name == "AvgPoolTransformation") {
        transformer.add<AvgPoolTransformation, opset1::AvgPool>(params);
        return;
    }
    if (name == "ClampTransformation") {
        transformer.add<ClampTransformation, opset1::Clamp>(params);
        return;
    }
    if (name == "ConvolutionTransformation" || name == "AsymmetricConvolutionTransformation") {
        transformer.add<ConvolutionTransformation, opset1::Convolution>(params);
        return;
    }
    if (name == "DepthToSpaceTransformation") {
        transformer.add<DepthToSpaceTransformation, opset1::DepthToSpace>(params);
        return;
    }
    if (name == "FakeQuantizeTransformation") {
        transformer.add<FakeQuantizeTransformation, opset1::FakeQuantize>(params);
        return;
    }
    if (name == "InterpolateTransformation") {
        transformer.add<InterpolateTransformation, ngraph::opset1::Interpolate>(params);
        return;
    }
    if (name == "MatMulTransformation") {
        transformer.add<MatMulTransformation, ngraph::opset1::MatMul>(params);
        return;
    }
    if (name == "MaxPoolTransformation") {
        transformer.add<MaxPoolTransformation, ngraph::opset1::MaxPool>(params);
        return;
    }
    if (name == "MultiplyTransformation") {
        transformer.add<MultiplyTransformation, ngraph::opset1::Multiply>(params);
        return;
    }
    if (name == "MVNTransformation") {
        transformer.add<MVNTransformation, ngraph::op::MVN>(params);
        return;
    }
    if (name == "NormalizeL2Transformation") {
        transformer.add<NormalizeL2Transformation, ngraph::opset1::NormalizeL2>(params);
        return;
    }
    if (name == "PReluTransformation") {
        transformer.add<PReluTransformation, ngraph::opset1::PRelu>(params);
        return;
    }
    if (name == "ReluTransformation") {
        transformer.add<ReluTransformation, ngraph::opset1::Relu>(params);
        return;
    }
    if (name == "ReshapeTransformation") {
        transformer.add<ReshapeTransformation, ngraph::opset1::Reshape>(params);
        return;
    }
    if (name == "SqueezeTransformation") {
        transformer.add<SqueezeTransformation, ngraph::opset1::Squeeze>(params);
        return;
    }
    if (name == "StridedSliceTransformation") {
        transformer.add<StridedSliceTransformation, ngraph::opset1::StridedSlice>(params);
        return;
    }
    if (name == "TransposeTransformation") {
        transformer.add<TransposeTransformation, ngraph::opset1::Transpose>(params);
        return;
    }
    if (name == "UnsqueezeTransformation") {
        transformer.add<UnsqueezeTransformation, ngraph::opset1::Unsqueeze>(params);
        return;
    }
    if (name == "FuseConvertTransformation") {
        transformer.add<FuseConvertTransformation, ngraph::opset1::Multiply>(params);
        return;
    }
    if (name == "FuseSubtractToFakeQuantizeTransformation") {
        transformer.add<FuseSubtractToFakeQuantizeTransformation, ngraph::opset1::Subtract>(params);
        return;
    }
    if (name == "FuseMultiplyToFakeQuantizeTransformation") {
        transformer.add<FuseMultiplyToFakeQuantizeTransformation, ngraph::opset1::Multiply>(params);
        return;
    }
    if (name == "MultiplyToGroupConvolutionTransformation") {
        transformer.add<MultiplyToGroupConvolutionTransformation, ngraph::opset1::Multiply>(params);
        return;
    }
    if (name == "SubtractMultiplyToMultiplyAddTransformation") {
        transformer.add<SubtractMultiplyToMultiplyAddTransformation, ngraph::opset1::Multiply>(params);
        return;
    }
    throw std::runtime_error("unexpected transformation name");
}

class TransformationsAfterSplitTransformation : public LayerTransformation, public testing::WithParamInterface<std::string> {
public:
    void SetUp() override {
        const auto layerName = GetParam();
        function = ngraph::builder::subgraph::TransformationsAfterSplitFunction::get(layerName);
        function->validate_nodes_and_infer_types();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        const auto layerName = obj.param;
        std::ostringstream result;

        result << "additional_layer_name_" << layerName;
        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> function;
};

TEST_P(TransformationsAfterSplitTransformation, Run) {
    const std::string layerName = GetParam();
    const auto params = LayerTransformation::createParamsU8I8();
    SimpleLowPrecisionTransformer transformer;
    getTransformerWithTransformationByName(transformer, params, layerName);

    ASSERT_NO_THROW(transformer.transform(function));
}

const std::vector<std::string> transformationNames = {
    "AddTransformationWithoutConcat",
    "AddTransformationWithConcat",
    "AvgPoolTransformation",
    "ClampTransformation",
    "ConvolutionTransformation",
    "AsymmetricConvolutionTransformation",
    "DepthToSpaceTransformation",
    "FakeQuantizeTransformation",
    "InterpolateTransformation",
    "MatMulTransformation",
    "MaxPoolTransformation",
    "MultiplyTransformation",
    "MVNTransformation",
    "NormalizeL2Transformation",
    "PReluTransformation",
    "ReluTransformation",
    "ReshapeTransformation",
    "SqueezeTransformation",
    "StridedSliceTransformation",
    "TransposeTransformation",
    "UnsqueezeTransformation",
    "FuseConvertTransformation",
    "FuseSubtractToFakeQuantizeTransformation",
    "FuseMultiplyToFakeQuantizeTransformation",
    "MultiplyToGroupConvolutionTransformation",
    "SubtractMultiplyToMultiplyAddTransformation",
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    TransformationsAfterSplitTransformation,
    ::testing::ValuesIn(transformationNames),
    TransformationsAfterSplitTransformation::getTestCaseName);

} // namespace
