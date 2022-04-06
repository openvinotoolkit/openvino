// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <sstream>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/weightable_layer_transformation.hpp>
#include "lpt_ngraph_functions/convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class IsAsymmetricOnWeightsDequantizationTestValues {
public:
    TestTransformationParams params;
    ngraph::element::Type precisionBeforeDequantization;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
    std::shared_ptr<ngraph::opset1::Constant> weights;
    builder::subgraph::DequantizationOperations dequantizationOnWeights;
    bool isAsymmetricOnWeights;
};

typedef std::tuple<
    element::Type,
    ngraph::PartialShape,
    IsAsymmetricOnWeightsDequantizationTestValues> IsAsymmetricOnWeightsDequantizationParams;

class IsAsymmetricOnWeightsDequantizationTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<IsAsymmetricOnWeightsDequantizationParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::getOriginal(
            netPrecision,
            testValues.precisionBeforeDequantization,
            inputShape,
            testValues.dequantizationOnActivations,
            testValues.weights,
            {},
            testValues.dequantizationOnWeights);
    }

    static std::string getTestCaseName(testing::TestParamInfo<IsAsymmetricOnWeightsDequantizationParams> obj) {
        const auto netPrecision = std::get<0>(obj.param);
        auto inputShape = std::get<1>(obj.param);
        IsAsymmetricOnWeightsDequantizationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            netPrecision << "_" <<
            inputShape << "_" <<
            testValues.precisionBeforeDequantization << "_" <<
            testValues.dequantizationOnActivations << "_" << "_weights_" <<
            testValues.weights->get_element_type() << "_" << "{ " <<
            testValues.weights->cast_vector<float>()[0] << " }_" <<
            testValues.dequantizationOnWeights;
        return result.str();
    }
};

TEST_P(IsAsymmetricOnWeightsDequantizationTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    const auto convolutions = LayerTransformation::get<opset1::Convolution>(actualFunction);
    ASSERT_TRUE(convolutions.size() == 1ul) << "convolution was not found";

    IsAsymmetricOnWeightsDequantizationTestValues testValues = std::get<2>(GetParam());

    const auto isAsymmetricOnWeights = ngraph::pass::low_precision::WeightableLayerTransformation::isAsymmetricOnWeights(convolutions[0],
        testValues.params.defaultPrecisions);
    ASSERT_EQ(testValues.isAsymmetricOnWeights, isAsymmetricOnWeights);
}

const std::vector<element::Type> netPrecisions = {
    element::f32
};

const std::vector<ngraph::PartialShape> suitablePartialShapes = {
    ngraph::PartialShape({ 1, 3, 72, 48 }),
    ngraph::PartialShape({ 4, 3, 72, 48 }),
    ngraph::PartialShape({ Dimension::dynamic(), 3, 72, 48 }),
    ngraph::PartialShape({ 1, 3, Dimension::dynamic(), Dimension::dynamic() }),
};

const std::vector<IsAsymmetricOnWeightsDequantizationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        ngraph::element::u8,
        {{ngraph::element::f32}, { 128.f }, { 0.02f }},
        op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
        {
            {ngraph::element::f32},
            {{1, 2, 3, 4, 5, 6}, ngraph::element::f32, {6, 1, 1, 1}},
            {{1, 2, 3, 4, 5, 6}, ngraph::element::f32, {6, 1, 1, 1}}
        },
        true
    },
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        ngraph::element::u8,
        {{ngraph::element::f32}, { 128.f }, { 0.02f }},
        op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
        {
            {ngraph::element::f32},
            {{0, 0, 1.e-7, 0, 0, 0}, ngraph::element::f32, {6, 1, 1, 1}},
            {{1, 2, 3, 4, 5, 6}, ngraph::element::f32, {6, 1, 1, 1}}
        },
        false
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    IsAsymmetricOnWeightsDequantizationTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(suitablePartialShapes),
        ::testing::ValuesIn(testValues)),
    IsAsymmetricOnWeightsDequantizationTransformation::getTestCaseName);
