// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/convolution_backprop_data.hpp>
#include <low_precision/network_helper.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/convolution_backprop_data_function.hpp"

namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ConvolutionBackpropDataTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        builder::subgraph::DequantizationOperations dequantizationOnWeights;
        std::shared_ptr<ngraph::opset1::Constant> weights;

        Actual() = default;
        Actual(
            const ngraph::element::Type& precisionBeforeDequantization,
            const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
            const builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
            const std::shared_ptr<ngraph::opset1::Constant>& weights) :
                precisionBeforeDequantization(precisionBeforeDequantization),
                dequantizationOnActivations(dequantizationOnActivations),
                fakeQuantizeOnWeights(fakeQuantizeOnWeights),
                weights(weights) {}
        Actual(
            const  ngraph::element::Type& precisionBeforeDequantization,
            const  ngraph::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
            const  builder::subgraph::DequantizationOperations& dequantizationOnWeights,
            const std::shared_ptr<ngraph::opset1::Constant>& weights) :
            precisionBeforeDequantization(precisionBeforeDequantization),
            dequantizationOnActivations(dequantizationOnActivations),
            dequantizationOnWeights(dequantizationOnWeights),
            weights(weights) {}
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        bool transformed;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
        element::Type,
        ngraph::PartialShape,
        ConvolutionBackpropDataTransformationTestValues> ConvolutionBackpropDataTransformationParams;

class ConvolutionBackpropDataTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionBackpropDataTransformationParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());

        ngraph::Shape outputShape;
        if (inputShape.is_static()) {
            outputShape = inputShape.to_shape();
        } else {
            outputShape = ngraph::Shape({ 1, 8, 16, 16 });
        }
        outputShape[1] /= 4;
        outputShape[2] *= 2;
        outputShape[3] *= 2;

        bool channelsIsDynamic = inputShape.rank().is_dynamic() || inputShape[1].is_dynamic();
        const size_t inputChannels = channelsIsDynamic ? 8ul : static_cast<size_t>(inputShape[1].get_length());

        std::shared_ptr<Node> actualWeights = pass::low_precision::fold<opset1::Broadcast>(
                testValues.actual.weights,
                opset1::Constant::create(
                        element::i64,
                        Shape{ outputShape.size() },
                        Shape{ inputChannels, outputShape[1], 1, 1 }));
        if (!testValues.actual.fakeQuantizeOnWeights.empty()) {
            actualWeights = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                    outputShape,
                    netPrecision,
                    testValues.actual.fakeQuantizeOnWeights,
                    as_type_ptr<opset1::Constant>(actualWeights));
        } else {
            actualWeights = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                    outputShape,
                    netPrecision,
                    testValues.actual.dequantizationOnWeights,
                    as_type_ptr<opset1::Constant>(actualWeights));
        }

        actualFunction = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getOriginal(
                testValues.actual.precisionBeforeDequantization,
                netPrecision,
                inputShape,
                outputShape,
                testValues.actual.dequantizationOnActivations,
                actualWeights);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionBackpropDataTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.transform(actualFunction);

        std::shared_ptr<Node> refWeights = pass::low_precision::fold<opset1::Broadcast>(
                testValues.expected.weights,
                opset1::Constant::create(
                        element::i64,
                        Shape{ outputShape.size() },
                        Shape{ inputChannels, outputShape[1], 1, 1 }));

        if (!testValues.expected.transformed) {
            refWeights = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                outputShape,
                netPrecision,
                testValues.actual.fakeQuantizeOnWeights,
                as_type_ptr<opset1::Constant>(refWeights));
        } else {
            refWeights = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                outputShape,
                netPrecision,
                testValues.expected.dequantizationOnWeights,
                as_type_ptr<opset1::Constant>(refWeights));
        }

        referenceFunction = ngraph::builder::subgraph::ConvolutionBackpropDataFunction::getReference(
                testValues.expected.precisionBeforeDequantization,
                netPrecision,
                inputShape,
                outputShape,
                testValues.expected.dequantizationOnActivations,
                refWeights,
                testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionBackpropDataTransformationParams> obj) {
        const auto netPrecision = std::get<0>(obj.param);
        auto inputShape = std::get<1>(obj.param);
        ConvolutionBackpropDataTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
               netPrecision << "_" <<
               inputShape << "_" <<
               testValues.actual.precisionBeforeDequantization << "_" <<
               testValues.actual.dequantizationOnActivations << "_" <<
               testValues.actual.dequantizationOnWeights << "_" <<
               testValues.actual.fakeQuantizeOnWeights << "_" <<"_weights_" <<
               testValues.actual.weights->get_element_type() << "_" << "{ " <<
               testValues.actual.weights->cast_vector<float>()[0] << " }_";
        return result.str();
    }
};

TEST_P(ConvolutionBackpropDataTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<element::Type> netPrecisions = {
    element::f32,
    element::f16
};

namespace testValues1 {
const std::vector<ngraph::PartialShape> shapes = {
    { 1, 8, 16, 16 },
    { Dimension::dynamic(), 8, Dimension::dynamic(), Dimension::dynamic() }
};

const std::vector<ConvolutionBackpropDataTransformationTestValues> testValues = {
    // with zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, {}, false }, {}},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // with zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            { 255ul, Shape({}), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, {}, false }, {}},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // updatePrecisions = false
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, {}, false }, {}},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            {{ngraph::element::f32}, { 2.f }, { 0.01f }},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, {}, false }, {}},
            {{}, { { 2.f }, ngraph::element::f32, {1, 2, 1, 1}, true, 1ul, element::i8, false, { "DISABLED_CONSTANT_FOLDING" }  }, {}},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({}), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {{ngraph::element::f32}, {}, { 0.01f }},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, {1}}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // per-channel dequantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { std::vector<float>{0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f}  }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // per-channel dequantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { std::vector<float>{0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f} }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { std::vector<float>{0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f} }},
            {},
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            true
        }
    },
    // per-channel dequantization on weights
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 1, 2, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 2, 1, 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {{ngraph::element::f32}, {}, { std::vector<float>{0.01f, 0.01f} }},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 2, 1, 1 }}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // issue #56886: unsupported per-batch dequantization on weights
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 8, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {{ngraph::element::f32}, {}, { std::vector<float>{0.01f}, ngraph::element::f32, {8, 1, 1, 1} }},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 0.02f }),
            true
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionBackpropDataTransformation,
    ::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(shapes),
    ::testing::ValuesIn(testValues)),
    ConvolutionBackpropDataTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> shapesWithDynamicChannels = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
    PartialShape::dynamic()
};

const std::vector<ConvolutionBackpropDataTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {{ngraph::element::f32}, {}, { 0.01f }},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 0.02f }),
            true
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionBackpropDataTransformation,
    ::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(shapesWithDynamicChannels),
    ::testing::ValuesIn(testValues)),
    ConvolutionBackpropDataTransformation::getTestCaseName);
} // namespace testValues2
} // namespace
