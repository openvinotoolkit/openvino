// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/network_helper.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/convolution_backprop_data.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

using const_node_ptr = const std::shared_ptr<const ov::Node>;
using callback_function_type = std::function<bool(const_node_ptr&)>;

bool empty_callback(const std::shared_ptr<const ov::Node>& node) {
    return false;
}

class ConvolutionBackpropDataTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        std::shared_ptr<ov::op::v0::Constant> weights;
        callback_function_type callback;

        Actual() = default;
        Actual(
            const ov::element::Type& precisionBeforeDequantization,
            const DequantizationOperations& dequantizationOnActivations,
            const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
            const DequantizationOperations& dequantizationOnWeights,
            const std::shared_ptr<ov::op::v0::Constant>& weights,
            const callback_function_type& callback = empty_callback) :
                precisionBeforeDequantization(precisionBeforeDequantization),
                dequantizationOnActivations(dequantizationOnActivations),
                fakeQuantizeOnWeights(fakeQuantizeOnWeights),
                dequantizationOnWeights(dequantizationOnWeights),
                weights(weights),
                callback(callback) {}
        Actual(
            const ov::element::Type& precisionBeforeDequantization,
            const ov::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
            const ov::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
            const std::shared_ptr<ov::op::v0::Constant>& weights,
            const callback_function_type& callback = empty_callback) :
                precisionBeforeDequantization(precisionBeforeDequantization),
                dequantizationOnActivations(dequantizationOnActivations),
                fakeQuantizeOnWeights(fakeQuantizeOnWeights),
                weights(weights),
                callback(callback) {}
        Actual(
            const ov::element::Type& precisionBeforeDequantization,
            const ov::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
            const ov::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
            const std::shared_ptr<ov::op::v0::Constant>& weights,
            const callback_function_type& callback = empty_callback) :
                precisionBeforeDequantization(precisionBeforeDequantization),
                dequantizationOnActivations(dequantizationOnActivations),
                dequantizationOnWeights(dequantizationOnWeights),
                weights(weights),
                callback(callback) {}
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
        std::shared_ptr<ov::op::v0::Constant> weights;
        bool transformed;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
        element::Type,
        ov::PartialShape,
        ConvolutionBackpropDataTransformationTestValues> ConvolutionBackpropDataTransformationParams;

class ConvolutionBackpropDataTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionBackpropDataTransformationParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());

        ov::Shape outputShape;
        if (inputShape.is_static()) {
            outputShape = inputShape.to_shape();
        } else {
            outputShape = ov::Shape({ 1, 8, 16, 16 });
        }
        outputShape[1] /= 4;
        outputShape[2] *= 2;
        outputShape[3] *= 2;

        bool channelsIsDynamic = inputShape.rank().is_dynamic() || inputShape[1].is_dynamic();
        const size_t inputChannels = channelsIsDynamic ? 8ul : static_cast<size_t>(inputShape[1].get_length());

        std::shared_ptr<Node> actualWeights = ov::pass::low_precision::fold<opset1::Broadcast>(
                testValues.actual.weights,
                ov::op::v0::Constant::create(
                        element::i64,
                        Shape{ outputShape.size() },
                        Shape{ inputChannels, outputShape[1], 1, 1 }));
        if (!testValues.actual.fakeQuantizeOnWeights.empty() && !testValues.actual.dequantizationOnWeights.empty()) {
            actualWeights = ConvolutionBackpropDataFunction::getWeights(
                    outputShape,
                    netPrecision,
                    testValues.actual.fakeQuantizeOnWeights,
                    testValues.actual.dequantizationOnWeights,
                    ov::as_type_ptr<ov::op::v0::Constant>(actualWeights));
        } else if (!testValues.actual.fakeQuantizeOnWeights.empty()) {
            actualWeights = ov::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                    outputShape,
                    netPrecision,
                    testValues.actual.fakeQuantizeOnWeights,
                    ov::as_type_ptr<ov::op::v0::Constant>(actualWeights));
        } else {
            actualWeights = ov::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                    outputShape,
                    netPrecision,
                    testValues.actual.dequantizationOnWeights,
                    ov::as_type_ptr<ov::op::v0::Constant>(actualWeights));
        }

        actualFunction = ov::builder::subgraph::ConvolutionBackpropDataFunction::getOriginal(
                testValues.actual.precisionBeforeDequantization,
                netPrecision,
                inputShape,
                outputShape,
                testValues.actual.dequantizationOnActivations,
                actualWeights);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::ConvolutionBackpropDataTransformation, opset1::ConvolutionBackpropData>(testValues.params);
        transform.get_pass_config()->set_callback<ov::pass::low_precision::ConvolutionBackpropDataTransformation>(testValues.actual.callback);
        transform.transform(actualFunction);

        std::shared_ptr<Node> refWeights = ov::pass::low_precision::fold<opset1::Broadcast>(
                testValues.expected.weights,
                ov::op::v0::Constant::create(
                        element::i64,
                        Shape{ outputShape.size() },
                        Shape{ inputChannels, outputShape[1], 1, 1 }));

        if (!testValues.expected.transformed) {
            refWeights = ov::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                outputShape,
                netPrecision,
                testValues.actual.fakeQuantizeOnWeights,
                testValues.actual.dequantizationOnWeights,
                ov::as_type_ptr<ov::op::v0::Constant>(refWeights));
        } else {
            refWeights = ov::builder::subgraph::ConvolutionBackpropDataFunction::getWeights(
                outputShape,
                netPrecision,
                testValues.expected.dequantizationOnWeights,
                ov::as_type_ptr<ov::op::v0::Constant>(refWeights));
        }

        referenceFunction = ov::builder::subgraph::ConvolutionBackpropDataFunction::getReference(
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<element::Type> netPrecisions = {
    element::f32,
    element::f16
};

namespace testValues1 {
const std::vector<ov::PartialShape> shapes = {
    { 1, 8, 16, 16 },
    { -1, -1, -1, -1 }
};

const std::vector<ConvolutionBackpropDataTransformationTestValues> testValues = {
    // with zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.02f }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, { { 128.f }, ov::element::f32, {}, false }, {}},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // with zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.02f }},
            { 255ul, Shape({}), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, { { 128.f }, ov::element::f32, {}, false }, {}},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // updatePrecisions = false
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.02f }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, { { 128.f }, ov::element::f32, {}, false }, {}},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.02f }},
            {{ov::element::f32}, { 2.f }, { 0.01f }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, { { 128.f }, ov::element::f32, {}, false }, {}},
            {{}, { { 2.f }, ov::element::f32, {1, 2, 1, 1}, true, 1ul, element::i8, false,
                   { {ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()} }  }, {}},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({}), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {{ov::element::f32}, {}, { 0.01f }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // mixed precision: f16 dequantization constants, f32 dequantization precision
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f16}, {}, { 0.02f }},
            {{ov::element::f16}, {}, { 0.03f }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0006f }, ov::element::f16, {}, false, 1, ov::element::f32}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // per-channel dequantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { std::vector<float>{0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f}  }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // per-channel dequantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { std::vector<float>{0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f} }},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { std::vector<float>{0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f, 0.02f, 0.01f} }},
            {},
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ -1.25f }),
            true
        }
    },
    // per-channel dequantization on weights
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 1, 2, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ -125.f }),
            true
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {{ov::element::f32}, {}, { std::vector<float>{0.01f, 0.01f} }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            {},
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f }),
            true
        }
    },
    // issue #56886: unsupported per-batch dequantization on weights
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 8, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ -1.25f }),
            true
        }
    },
    // FQ + DQ version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            { 255ul, Shape({ 8, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f }, ov::element::i8 },
            {{ov::element::f32}, {}, { {0.01f}, ov::element::f32, {8, 1, 1, 1} }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f }),
            false
        }
    },
    // QDq version
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {{ov::element::f32}, {}, { std::vector<float>{0.01f}, ov::element::f32, {8, 1, 1, 1} }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ 0.02f }),
            true
        }
    },
    //  issue #59593: subtract on activations, non-asymmetric
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.01f}},
            { 255ul, Shape({ 1, 2, 1, 1 }), { 0.f }, { 254.f }, { 0.f }, { 25.4f } },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f }),
            [](const_node_ptr& node) {
                return ov::pass::low_precision::LayerTransformation::isAsymmetricQuantization(node,
                    { ov::element::u8,  ov::element::i8 });
            }
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.01f}},
            {},
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ 2.f }),
            false // weights are not folded because of callback returning true
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
const std::vector<ov::PartialShape> shapesWithDynamicChannels = {
    PartialShape::dynamic()
};

const std::vector<ConvolutionBackpropDataTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {{ov::element::f32}, {}, { 0.01f }},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{ 2.f })
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.02f }},
            {},
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{ 0.02f }),
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
