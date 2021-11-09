// Copyright (C) 2021 Intel Corporation
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

class IsAsymmetricOnWeightsTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    TestTransformationParams params;
    Actual actual;
};

typedef std::tuple<
    element::Type,
    ngraph::PartialShape,
    IsAsymmetricOnWeightsTestValues,
    std::pair<std::vector<bool>, bool> > IsAsymmetricOnWeightsParams;

class IsAsymmetricOnWeightsTransformation : public LayerTransformation, public testing::WithParamInterface<IsAsymmetricOnWeightsParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());
        std::pair<std::vector<bool>, bool> transposeAndIsAsymmetricOnWeights = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::getOriginal(
            netPrecision,
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            testValues.actual.dequantizationOnActivations,
            testValues.actual.weights,
            testValues.actual.fakeQuantizeOnWeights,
            transposeAndIsAsymmetricOnWeights.first[0],
            transposeAndIsAsymmetricOnWeights.first[1],
            transposeAndIsAsymmetricOnWeights.first[2],
            transposeAndIsAsymmetricOnWeights.first[3],
            transposeAndIsAsymmetricOnWeights.first[4]);
    }

    static std::string getTestCaseName(testing::TestParamInfo<IsAsymmetricOnWeightsParams> obj) {
        const auto netPrecision = std::get<0>(obj.param);
        auto inputShape = std::get<1>(obj.param);
        IsAsymmetricOnWeightsTestValues testValues = std::get<2>(obj.param);
        std::pair<std::vector<bool>, bool> transposeAndIsAsymmetricOnWeights = std::get<3>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            netPrecision << "_" <<
            inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantizationOnActivations << "_" << "_weights_" <<
            testValues.actual.weights->get_element_type() << "_" << "{ " <<
            testValues.actual.weights->cast_vector<float>()[0] << " }_" <<
            testValues.actual.fakeQuantizeOnWeights << "_" <<
            transposeAndIsAsymmetricOnWeights.first[0] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[1] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[2] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[3] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[4];
        return result.str();
    }
};

TEST_P(IsAsymmetricOnWeightsTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    const auto convolutions = LayerTransformation::get<opset1::Convolution>(actualFunction);
    ASSERT_TRUE(convolutions.size() == 1ul) << "convolution was not found";

    const auto isAsymmetricOnWeights = ngraph::pass::low_precision::WeightableLayerTransformation::isAsymmetricOnWeights(convolutions[0]);
    std::pair<std::vector<bool>, bool> transposeAndIsAsymmetricOnWeights = std::get<3>(GetParam());
    ASSERT_EQ(transposeAndIsAsymmetricOnWeights.second, isAsymmetricOnWeights);
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

const std::vector<IsAsymmetricOnWeightsTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.f }, { 1.27f } },
        }
    }
};

const std::vector<std::pair<std::vector<bool>, bool> > transposeFlags = {
    // asymmetric quantization
    {{false, false, false, false, false}, true},
    {{true, false, false, false, false}, true},

    // not supported FakeQuantize
    {{false, true, false, false, false}, false},
    {{false, false, true, false, false}, false},
    {{false, false, false, true, false}, false},
    {{false, false, false, false, true}, false}
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    IsAsymmetricOnWeightsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(suitablePartialShapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(transposeFlags)),
    IsAsymmetricOnWeightsTransformation::getTestCaseName);
