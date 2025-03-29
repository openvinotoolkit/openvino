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
#include "low_precision/multiply_to_group_convolution.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/multiply_to_group_convolution.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class MultiplyToGroupConvolutionTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        std::shared_ptr<ov::op::v0::Constant> weights;
        std::shared_ptr<ov::op::v0::Constant> biases;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    ov::PartialShape inputShape;
    TestTransformationParams params;
    bool transformed;
    bool haveMultiplyWithNoConstBeforeDequantization;
    Actual actual;
    Expected expected;
};

class MultiplyToGroupConvolutionTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<MultiplyToGroupConvolutionTransformationTestValues> {
public:
    void SetUp() override {
        const MultiplyToGroupConvolutionTransformationTestValues testValues = GetParam();

        actualFunction = ov::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.haveMultiplyWithNoConstBeforeDequantization);

        auto precisionRestrictions = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
            ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Multiply>({
                {{0}, {ov::element::u8}},
                {{1}, {ov::element::i8}}
            })
        });

        SimpleLowPrecisionTransformer transformer(precisionRestrictions);
        transformer.add<ov::pass::low_precision::MultiplyToGroupConvolutionTransformation, ov::op::v1::Multiply>(testValues.params);
        transformer.transform(actualFunction);

        if (testValues.transformed) {
            referenceFunction = ov::builder::subgraph::MultiplyToGroupConvolutionFunction::getReference(
                testValues.inputShape,
                testValues.expected.inputPrecision,
                testValues.expected.weights,
                testValues.expected.biases,
                testValues.expected.dequantization);
        } else {
            referenceFunction = ov::builder::subgraph::MultiplyToGroupConvolutionFunction::getOriginal(
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

TEST_P(MultiplyToGroupConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<MultiplyToGroupConvolutionTransformationTestValues> testValues = {
    // only multiply
    {
        { 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            nullptr,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // only multiply with dynamic shape
    {
        { Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic() },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            nullptr,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // only multiply with dynamic shape (dynamic channels)
    {
        { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {}
    },
    // subtract + multiply
    {
        { 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{-0.77f, 0.8f, 0.1f, 1.5f}},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1, 4, 1, 1}, std::vector<float>{0.77f, -0.8f, -0.1f, -1.5f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // subtract + multiply with dynamic channels
    {
        { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{-0.77f, 0.8f, 0.1f, 1.5f}},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1, 4, 1, 1}, std::vector<float>{0.77f, -0.8f, -0.1f, -1.5f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // subtract + multiply with dynamic rank (not transformed)
    {
        PartialShape::dynamic(),
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{-0.77f, 0.8f, 0.1f, 1.5f}},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {}
    },
    // subtract + multiply
    {
        { 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4, 1, 1}, true, 1, ov::element::u8, true},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1, 4, 1, 1}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // without convert
    {
        { 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {},
                DequantizationOperations::Subtract{{1.f, 2.f, 3.f, 4.f}, element::f32}.setConstantPrecision(element::f32),
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1, 4, 1, 1}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // 5d
    {
        { 1, 4, 1, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            ov::element::u8,
            {
                {},
                DequantizationOperations::Subtract{{1.f, 2.f, 3.f, 4.f}, element::f32}.setConstantPrecision(element::f32),
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1, 4, 1, 1, 1}, std::vector<float>{-1.f, -2.f, -3.f, -4.f}),
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    // TODO: LPT: not implemented
//    // i8 (not transformed)
//    {
//        ov::Shape{ 1, 4, 1, 1 },
//        LayerTransformation::createParamsU8I8(),
//        false,
//        false,
//        {
//            ov::element::i8,
//            {
//                {},
//                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32},
//                {{0.45f, 0.82f, 0.71f, 0.37f}}
//            }
//        },
//        {}
//    },
    // by spatial dimensions (not transformed)
    {
        { 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ov::element::u8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32,  { 1, 1, 2, 2 }},
                {{0.45f, 0.82f, 0.71f, 0.37f}, ov::element::f32,  { 1, 1, 2, 2 }}
            }
        },
        {}
    },
    // 3d (not transformed)
    {
        { 1, 4, 1 },
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            ov::element::u8,
            {
                {},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, { 1, 4, 1 }},
                {{0.45f, 0.82f, 0.71f, 0.37f}, ov::element::f32, { 1, 4, 1 }}
            }
        },
        {}
    },
    {
        { 1, 4, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        false,
        true,
        {
            ov::element::u8,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            nullptr,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    },
    {
        { Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic() },
        LayerTransformation::createParamsU8I8(),
        false,
        true,
        {
            ov::element::u8,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        },
        {
            ov::element::u8,
            std::make_shared<ov::op::v0::Constant>(ov::element::i8, Shape{4, 1, 1, 1, 1}, std::vector<float>{1.f, 1.f, 1.f, 1.f}),
            nullptr,
            {
                {},
                {},
                {{0.45f, 0.82f, 0.71f, 0.37f}}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MultiplyToGroupConvolutionTransformation,
    ::testing::ValuesIn(testValues),
    MultiplyToGroupConvolutionTransformation::getTestCaseName);
