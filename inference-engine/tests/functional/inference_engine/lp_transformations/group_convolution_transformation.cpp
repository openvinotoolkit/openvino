// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/group_convolution.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/group_convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class GroupConvolutionTransformationTestParams {
public:
    low_precision::LayerTransformation::Params transformationParams;
    ngraph::Shape inputShape;
    ngraph::Shape outputShape;
    size_t group;
    ngraph::builder::subgraph::GroupConvolutionFunction::ActualValues actual;
    ngraph::builder::subgraph::GroupConvolutionFunction::ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    bool,
    GroupConvolutionTransformationTestParams> ConvolutionTransformationParams;

class GroupConvolutionTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const bool updatePrecisions = std::get<1>(GetParam());
        const GroupConvolutionTransformationTestParams testValues = std::get<2>(GetParam());

        const low_precision::LayerTransformation::Params params = low_precision::LayerTransformation::Params(testValues.transformationParams).
            setUpdatePrecisions(updatePrecisions);

        actualFunction = ngraph::builder::subgraph::GroupConvolutionFunction::getOriginal(
            precision,
            testValues.inputShape,
            testValues.outputShape,
            testValues.group,
            params.updatePrecisions,
            testValues.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::GroupConvolutionTransformation, ngraph::opset1::GroupConvolution>(params);
        transform.transform(actualFunction);

        referenceFunction = testValues.expected.activationPrecision == ngraph::element::f32 ?
            ngraph::builder::subgraph::GroupConvolutionFunction::getOriginal(
                precision,
                testValues.inputShape,
                testValues.outputShape,
                testValues.group,
                params.updatePrecisions,
                testValues.actual) :
            ngraph::builder::subgraph::GroupConvolutionFunction::getReference(
                precision,
                testValues.inputShape,
                testValues.outputShape,
                testValues.group,
                params.updatePrecisions,
                testValues.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
        ngraph::element::Type precision;
        bool updatePrecisions;
        GroupConvolutionTransformationTestParams params;
        std::tie(precision, updatePrecisions, params) = obj.param;

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, params.inputShape, params.transformationParams.setUpdatePrecisions(updatePrecisions)) <<
            params.outputShape << "_" <<
            params.group << "_" <<
            params.actual << "_" <<
            params.expected;
        return result.str();
    }
};

TEST_P(GroupConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<bool> updatePrecisions = {
    true,
    false
};

const std::vector<GroupConvolutionTransformationTestParams> testParams = {
    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        { 1, 6, 224, 224 },
        { 1, 24, 218, 218 },
        3ul,
        // ActualValues
        {
            ngraph::element::u8,
            { 128 },
            { 0.02f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 128 },
            ngraph::element::i8,
            { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
            { },
            std::vector<float>(24, 0.0002f)  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },
    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        { 1, 6, 224, 224 },
        { 1, 24, 218, 218 },
        3ul,
        // ActualValues
        {
            ngraph::element::u8,
            { 128 },
            { 0.02f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            { },
            ngraph::element::f32,
            { },
            { },
            { }
        }
    },
    // group convolution, per-channel quantization, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 6, 224, 224 },
        { 1, 24, 218, 218 },
        3ul,
        // ActualValues
        {
            ngraph::element::u8,
            { },
            { 0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { },
            ngraph::element::i8,
            { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
            { },
            {
                // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f,
                // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f,
                // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f
            }
        }
    },
    // depth-wise convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 6, 224, 224 },
        { 1, 6, 218, 218 },
        3ul,
        // ActualValues
        {
            ngraph::element::u8,
            { 128 },
            { 0.02f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 128 },
            ngraph::element::i8,
            { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
            { },
            std::vector<float>(6, 0.0002f)  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },
    // depth-wise convolution, per-channel quantization, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 6, 224, 224 },
        { 1, 6, 218, 218 },
        6ul,
        // ActualValues
        {
            ngraph::element::u8,
            { },
            { 0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { },
            ngraph::element::i8,
            { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
            { },
            {
                // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                0.0002f, 0.0002f,
                // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                0.0004f, 0.0004f,
                // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                0.0008f, 0.0008f
            }
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(testParams)),
    GroupConvolutionTransformation::getTestCaseName);
