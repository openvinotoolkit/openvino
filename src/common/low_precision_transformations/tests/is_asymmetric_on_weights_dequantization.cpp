// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/weightable_layer_transformation.hpp"
#include <memory>
#include <sstream>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include <utility>

#include "layer_transformation.hpp"
#include "ov_lpt_models/convolution.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class IsAsymmetricOnWeightsDequantizationTestValues {
public:
    TestTransformationParams params;
    ov::element::Type precisionBeforeDequantization;
    ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
    std::shared_ptr<ov::op::v0::Constant> weights;
    ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    bool isAsymmetricOnWeights;
};

typedef std::tuple<element::Type, ov::PartialShape, IsAsymmetricOnWeightsDequantizationTestValues>
    IsAsymmetricOnWeightsDequantizationParams;

class IsAsymmetricOnWeightsDequantizationTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<IsAsymmetricOnWeightsDequantizationParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());

        actualFunction =
            ov::builder::subgraph::ConvolutionFunction::getOriginal(netPrecision,
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
        result << toString(testValues.params) << "_" << netPrecision << "_" << inputShape << "_"
               << testValues.precisionBeforeDequantization << "_" << testValues.dequantizationOnActivations << "_"
               << "_weights_" << testValues.weights->get_element_type() << "_"
               << "{ " << testValues.weights->cast_vector<float>()[0] << " }_" << testValues.dequantizationOnWeights;
        return result.str();
    }
};

TEST_P(IsAsymmetricOnWeightsDequantizationTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    const auto convolutions = LayerTransformation::get<opset1::Convolution>(actualFunction);
    ASSERT_TRUE(convolutions.size() == 1ul) << "convolution was not found";

    IsAsymmetricOnWeightsDequantizationTestValues testValues = std::get<2>(GetParam());

    const auto isAsymmetricOnWeights =
        ov::pass::low_precision::WeightableLayerTransformation::isAsymmetricOnWeights(
            convolutions[0],
            testValues.params.defaultPrecisions);
    ASSERT_EQ(testValues.isAsymmetricOnWeights, isAsymmetricOnWeights);
}

const std::vector<element::Type> netPrecisions = {element::f32};

const std::vector<ov::PartialShape> suitablePartialShapes = {
    ov::PartialShape({1, 3, 72, 48}),
    ov::PartialShape({4, 3, 72, 48}),
    ov::PartialShape({Dimension::dynamic(), 3, 72, 48}),
    ov::PartialShape({1, 3, Dimension::dynamic(), Dimension::dynamic()}),
};

const std::vector<IsAsymmetricOnWeightsDequantizationTestValues> testValues = {
    {LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
     ov::element::u8,
     {{ov::element::f32}, {128.f}, {0.02f}},
     op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{2.f}),
     {{ov::element::f32},
      {{1, 2, 3, 4, 5, 6}, ov::element::f32, {6, 1, 1, 1}},
      {{1, 2, 3, 4, 5, 6}, ov::element::f32, {6, 1, 1, 1}}},
     true},
    {LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
     ov::element::u8,
     {{ov::element::f32}, {128.f}, {0.02f}},
     op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{2.f}),
     {{ov::element::f32},
      {{0, 0, 1.e-7f, 0, 0, 0}, ov::element::f32, {6, 1, 1, 1}},
      {{1, 2, 3, 4, 5, 6}, ov::element::f32, {6, 1, 1, 1}}},
     false}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         IsAsymmetricOnWeightsDequantizationTransformation,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(suitablePartialShapes),
                                            ::testing::ValuesIn(testValues)),
                         IsAsymmetricOnWeightsDequantizationTransformation::getTestCaseName);
