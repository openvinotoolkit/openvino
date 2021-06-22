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
#include "low_precision/multiply_to_group_convolution.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/multiply_to_group_convolution_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class MultiplyToGroupConvolutionTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        std::shared_ptr<ngraph::opset1::Constant> biases;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool transformed;
    bool haveMultiplyWithNoConstBeforeDequantization;
    Actual actual;
    Expected expected;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class MultiplyToGroupConvolutionTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<MultiplyToGroupConvolutionTransformationTestValues> {
public:
    void SetUp() override {
        const MultiplyToGroupConvolutionTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.haveMultiplyWithNoConstBeforeDequantization);
        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation, ngraph::opset1::Multiply>(testValues.params);
        transformer.transform(actualFunction);

        if (testValues.transformed) {
            referenceFunction = ngraph::builder::subgraph::MultiplyToGroupConvolutionFunction::getReference(
                testValues.inputShape,
                testValues.expected.inputPrecision,
                testValues.expected.weights,
                testValues.expected.biases,
                testValues.expected.dequantization);
        } else {
            referenceFunction = ngraph::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
                testValues.inputShape,
                testValues.actual.precisionBeforeDequantization,
                testValues.actual.dequantization,
                testValues.haveMultiplyWithNoConstBeforeDequantization);
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<MultiplyToGroupConvolutionTransformationTestValues> obj) {
        const MultiplyToGroupConvolutionTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.transformed << "_" <<
            testValues.haveMultiplyWithNoConstBeforeDequantization << "_" <<
            testValues.actual.dequantization;
        return result.str();
    }
};

const std::vector<MultiplyToGroupConvolutionTransformationTestValues> testValues = {
    // only multiply
    {
        ngraph::Shape{ 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ngraph::element::u8,
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::i8, ngraph::Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            nullptr,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // subtract + multiply
    {
        ngraph::Shape{ 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{-0.77f, 0.8f, 0.1f, 1.5f}},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ngraph::element::u8,
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::i8, ngraph::Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{1, 4, 1, 1}, std::vector<float>{0.77f, -0.8f, -0.1f, -1.5f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // subtract + multiply
    {
        ngraph::Shape{ 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {1, 4, 1, 1}, true, 1, ngraph::element::u8, true},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ngraph::element::u8,
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::i8, ngraph::Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{1, 4, 1, 1}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // without convert
    {
        ngraph::Shape{ 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ngraph::element::u8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ngraph::element::u8,
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::i8, ngraph::Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{1, 4, 1, 1}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // 5d
    {
        ngraph::Shape{ 1, 4, 1, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ngraph::element::u8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ngraph::element::u8,
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::i8, ngraph::Shape{4, 1, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{1, 4, 1, 1, 1}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // i8 (not transformed)
    {
        ngraph::Shape{ 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ngraph::element::i8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {}
    },
    // by spatial dimensions (not transformed)
    {
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ngraph::element::u8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32,  ngraph::Shape{ 1, 1, 2, 2 }},
                {{0.45f, 0.82f, 0.71f, 0.37f}, ngraph::element::f32,  ngraph::Shape{ 1, 1, 2, 2 }}
            }
        },
        {}
    },
    // 3d (not transformed)
    {
        ngraph::Shape{ 1, 4, 1 },
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ngraph::element::u8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, ngraph::Shape{ 1, 4, 1 }},
                {{0.45f, 0.82f, 0.71f, 0.37f}, ngraph::element::f32, ngraph::Shape{ 1, 4, 1 }}
            }
        },
        {}
    },
    {
        ngraph::Shape{ 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        false,
        true,
        {
            ngraph::element::u8,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ngraph::element::u8,
            std::make_shared<ngraph::opset1::Constant>(ngraph::element::i8, ngraph::Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            nullptr,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
};

TEST_P(MultiplyToGroupConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MultiplyToGroupConvolutionTransformation,
    ::testing::ValuesIn(testValues),
    MultiplyToGroupConvolutionTransformation::getTestCaseName);
