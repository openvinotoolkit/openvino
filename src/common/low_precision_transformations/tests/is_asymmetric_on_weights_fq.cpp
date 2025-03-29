// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <sstream>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/weightable_layer_transformation.hpp"
#include "ov_lpt_models/convolution.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class IsAsymmetricOnWeightsFakeQuantizeTestValues {
public:
    TestTransformationParams params;
    ov::element::Type precisionBeforeDequantization;
    ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
    std::shared_ptr<ov::op::v0::Constant> weights;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

typedef std::tuple<
    element::Type,
    ov::PartialShape,
    IsAsymmetricOnWeightsFakeQuantizeTestValues,
    std::pair<std::vector<bool>, bool> > IsAsymmetricOnWeightsFakeQuantizeParams;

class IsAsymmetricOnWeightsFakeQuantizeTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<IsAsymmetricOnWeightsFakeQuantizeParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());
        std::pair<std::vector<bool>, bool> transposeAndIsAsymmetricOnWeights = std::get<3>(GetParam());

        actualFunction = ov::builder::subgraph::ConvolutionFunction::getOriginal(
            netPrecision,
            testValues.precisionBeforeDequantization,
            inputShape,
            testValues.dequantizationOnActivations,
            testValues.weights,
            testValues.fakeQuantizeOnWeights,
            {},
            transposeAndIsAsymmetricOnWeights.first[0],
            transposeAndIsAsymmetricOnWeights.first[1],
            transposeAndIsAsymmetricOnWeights.first[2],
            transposeAndIsAsymmetricOnWeights.first[3],
            transposeAndIsAsymmetricOnWeights.first[4]);
    }

    static std::string getTestCaseName(testing::TestParamInfo<IsAsymmetricOnWeightsFakeQuantizeParams> obj) {
        const auto netPrecision = std::get<0>(obj.param);
        auto inputShape = std::get<1>(obj.param);
        IsAsymmetricOnWeightsFakeQuantizeTestValues testValues = std::get<2>(obj.param);
        std::pair<std::vector<bool>, bool> transposeAndIsAsymmetricOnWeights = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            netPrecision << "_" <<
            inputShape << "_" <<
            testValues.precisionBeforeDequantization << "_" <<
            testValues.dequantizationOnActivations << "_" << "_weights_" <<
            testValues.weights->get_element_type() << "_" << "{ " <<
            testValues.weights->cast_vector<float>()[0] << " }_" <<
            testValues.fakeQuantizeOnWeights << "_" <<
            transposeAndIsAsymmetricOnWeights.first[0] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[1] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[2] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[3] << "_" <<
            transposeAndIsAsymmetricOnWeights.first[4];
        return result.str();
    }
};

TEST_P(IsAsymmetricOnWeightsFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    const auto convolutions = LayerTransformation::get<opset1::Convolution>(actualFunction);
    ASSERT_TRUE(convolutions.size() == 1ul) << "convolution was not found";

    auto defaultPrecisions = std::get<2>(GetParam()).params.defaultPrecisions;
    const auto isAsymmetricOnWeights = ov::pass::low_precision::WeightableLayerTransformation::isAsymmetricOnWeights(convolutions[0],
        defaultPrecisions);
    std::pair<std::vector<bool>, bool> transposeAndIsAsymmetricOnWeights = std::get<3>(GetParam());
    ASSERT_EQ(transposeAndIsAsymmetricOnWeights.second, isAsymmetricOnWeights);
}

const std::vector<element::Type> netPrecisions = {
    element::f32
};

const std::vector<ov::PartialShape> suitablePartialShapes = {
    ov::PartialShape({ 1, 3, 72, 48 }),
    ov::PartialShape({ 4, 3, 72, 48 }),
    ov::PartialShape({ Dimension::dynamic(), 3, 72, 48 }),
    ov::PartialShape({ 1, 3, Dimension::dynamic(), Dimension::dynamic() }),
};

const std::vector<IsAsymmetricOnWeightsFakeQuantizeTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        ov::element::u8,
        {{ov::element::f32}, { 128.f }, { 0.02f }},
        op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ 2.f }),
        { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.f }, { 1.27f } },
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
    IsAsymmetricOnWeightsFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(suitablePartialShapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(transposeFlags)),
    IsAsymmetricOnWeightsFakeQuantizeTransformation::getTestCaseName);
