// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/group_convolution.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/group_convolution_function.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;


class GroupConvolutionTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        ngraph::element::Type precisionAfterDequantization;
    };

    TestTransformationParams params;
    size_t group;
    int groupCalculationDimention;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::pair<ngraph::PartialShape, ngraph::PartialShape>, // inputShape - outputShape
    GroupConvolutionTestValues> ConvolutionTransformationParams;

class GroupConvolutionTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam()).first;
        const auto outputShape = std::get<0>(GetParam()).second;
        const GroupConvolutionTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::GroupConvolutionFunction::get(
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            outputShape,
            testValues.group,
            testValues.groupCalculationDimention,
            testValues.actual.dequantization,
            testValues.actual.weights,
            testValues.actual.fakeQuantizeOnWeights,
            testValues.actual.dequantizationOnWeights,
            ngraph::element::f32,
            {},
            ngraph::element::f32);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::GroupConvolutionTransformation, ngraph::opset1::GroupConvolution>(testValues.params);
        if (testValues.params.supportAsymmetricQuantization == false) {
            transform.set_callback<ngraph::pass::low_precision::GroupConvolutionTransformation>(
                [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
                    return ngraph::pass::low_precision::LayerTransformation::isAsymmetricQuantization(node);
                });
        }
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::GroupConvolutionFunction::get(
            testValues.expected.precisionBeforeDequantization,
            inputShape,
            outputShape,
            testValues.group,
            testValues.groupCalculationDimention,
            testValues.expected.dequantizationBefore,
            testValues.expected.weights,
            testValues.expected.fakeQuantizeOnWeights,
            testValues.expected.dequantizationOnWeights,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            testValues.expected.precisionAfterDequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
        const auto inputShape = std::get<0>(obj.param).first;
        const auto outputShape = std::get<0>(obj.param).second;
        const GroupConvolutionTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            inputShape << "_" <<
            outputShape << "_" <<
            testValues.group << "_" <<
            testValues.groupCalculationDimention << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" << "_weights_" <<
            testValues.actual.weights->get_element_type() << "_" << "{ " <<
            testValues.actual.weights->cast_vector<float>()[0] << " }_" <<
            testValues.actual.fakeQuantizeOnWeights << "_";
        return result.str();
    }
};

TEST_P(GroupConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<std::pair<ngraph::PartialShape, ngraph::PartialShape>> shapesForGroupConv = {
    {{ 1, 6, 224, 224 }, { 1, 24, 218, 218 }},
    {
        { Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic() },
        { Dimension::dynamic(), 24, Dimension::dynamic(), Dimension::dynamic() }
    }
};

const std::vector<GroupConvolutionTestValues> testValuesGroupConv = {
    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 24, 1, 1 }}} // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        3ul,
        0,
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 24, 1, 1 }}} // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        3ul,
        1,
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 24, 1, 1 }}} // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ ngraph::element::f32 }, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {},
            ngraph::element::f32,
            {}
        }
    },
    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::f32,
            {{}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 24, 1, 1 }}} // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },
    // group convolution, per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{ 0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f
                    },
                    ngraph::element::f32, {1, 24, 1, 1}
                }
            },
        }
    },
    // group convolution, per-channel quantization with the same values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{ 0.02f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {},
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {
                {},
                {},
                {{ 0.0002f }, ngraph::element::f32, {1, 24, 1, 1}}
            },
        }
    },
    // group convolution, without zero point, without convert
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::f32,
            {{}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {{}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            {},
            ngraph::element::f32,
            {}
        }
    },
    // group convolution, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {{element::f32}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 24, 1, 1 }}}
        }
    },
    // per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{ 0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            {},
            { ngraph::element::f32, {}, {0.01}}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            {},
            {},
            ngraph::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f
                    },
                    ngraph::element::f32, {1, 24, 1, 1}
                }
            },
        }
    },

    // per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{255}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true},
                {{ 0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            {},
            {
                ngraph::element::f32,
                {{127}, ngraph::element::f32, {}, true, 1, ngraph::element::i8, true},
                {0.01}
            }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {},
                {std::vector<float>(6ul, 255.f), ngraph::element::f32, {1, 6, 1, 1}, false, 1, ngraph::element::u8},
                {}
            },
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ 2.f }),
            {},
            {
                {},
                {
                    std::vector<float>(24ul, 127.f),
                    ngraph::element::f32,
                    {24, 1, 1, 1},
                    false,
                    1,
                    ngraph::element::i8,
                    false,
                    {"DEQUANTIZATION", "DISABLED_CONSTANT_FOLDING"}
                },
                {}
            },
            ngraph::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f, 0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0008f
                    },
                    ngraph::element::f32, {1, 24, 1, 1}
                }
            },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesForGroupConv),
        ::testing::ValuesIn(testValuesGroupConv)),
    GroupConvolutionTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<std::pair<ngraph::PartialShape, ngraph::PartialShape>> shapesForDepthWiseConv = {
    {{ 1, 6, 224, 224 }, { 1, 6, 218, 218 }},
    {
        { Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic() },
        { Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic() }
    }
};

const std::vector<GroupConvolutionTestValues> testValuesForDepthWiseConv = {
    // depth-wise convolution, per-tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 6, 1, 1 }}}
        }
    },
    // depth-wise convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::f32,
            {{}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 6, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 6, 1, 1 }}}
        }
    },
    // depth-wise convolution, per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{ 0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {
                {},
                {},
                {
                    {
                        0.0002f, 0.0002f,  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0004f, 0.0004f,  // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0008f, 0.0008f   // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                    },
                    ngraph::element::f32, {1, 6, 1, 1}
                }
            },
        }
    },
    // depth-wise convolution, per-tensor quantization with the same values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{ 0.02f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {
                {},
                {},
                {{ 0.0002f }, ngraph::element::f32, {1, 6, 1, 1}}
            },
        }
    },
    // depth-wise convolution, without zero point, without convert
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        // ActualValues
        {
            ngraph::element::f32,
            {{}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {{}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            {},
            ngraph::element::f32,
            {}
        }
    },
    // depth-wise convolution, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {{element::f32}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 6, 1, 1 }}}
        }
    },
    // without dequantization operations
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        // ActualValues
        {
            ngraph::element::f32,
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesForDepthWiseConv),
        ::testing::ValuesIn(testValuesForDepthWiseConv)),
    GroupConvolutionTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<std::pair<ngraph::PartialShape, ngraph::PartialShape>> shapesWithDynamicChannel = {
    {
        { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
        { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() }
    },
    { PartialShape::dynamic(), PartialShape::dynamic() }
};

const std::vector<GroupConvolutionTestValues> testValuesWithDynamicChannel = {
    // depth-wise convolution, per-tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            {},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    GroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesWithDynamicChannel),
        ::testing::ValuesIn(testValuesWithDynamicChannel)),
    GroupConvolutionTransformation::getTestCaseName);
} // namespace testValues3
