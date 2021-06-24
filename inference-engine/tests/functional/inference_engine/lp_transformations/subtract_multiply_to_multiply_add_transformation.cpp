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
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/subtract_multiply_to_multiply_add_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class SubtractMultiplyToMultiplyAddTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBefore;
        DequantizationOperations dequantization;
        ngraph::element::Type precisionAfter;
    };
    class Expected {
    public:
        ngraph::element::Type precisionBefore;
        DequantizationOperations dequantization;
        ngraph::element::Type precisionAfter;
        Multiply multiply;
        Add add;
    };
    ngraph::Shape shape;
    low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class SubtractMultiplyToMultiplyAddTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<SubtractMultiplyToMultiplyAddTransformationTestValues> {
public:
    void SetUp() override {
        SubtractMultiplyToMultiplyAddTransformationTestValues testValues = GetParam();

        actualFunction = SubtractMultiplyToMultiplyAddFunction::getOriginal(
            testValues.shape,
            testValues.actual.precisionBefore,
            testValues.actual.dequantization,
            testValues.actual.precisionAfter);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::SubtractMultiplyToMultiplyAddTransformation, ngraph::opset1::Multiply>(
            low_precision::LayerTransformation::Params(testValues.params));
        transform.transform(actualFunction);

        referenceFunction = SubtractMultiplyToMultiplyAddFunction::getReference(
            testValues.shape,
            testValues.expected.precisionBefore,
            testValues.expected.dequantization,
            testValues.expected.precisionAfter,
            testValues.expected.multiply,
            testValues.expected.add);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SubtractMultiplyToMultiplyAddTransformationTestValues> obj) {
        SubtractMultiplyToMultiplyAddTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.actual.precisionBefore << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.precisionAfter << "_" <<
            testValues.expected.precisionBefore << "_" <<
            testValues.expected.dequantization << "_" <<
            testValues.expected.precisionAfter << "_" <<
            testValues.expected.multiply.values << "_" <<
            testValues.expected.multiply.constantShape << "_" <<
            testValues.expected.multiply.outPrecision << "_" <<
            testValues.expected.add.values << "_" <<
            testValues.expected.add.constantShape << "_" <<
            testValues.expected.add.outPrecision;
        return result.str();
    }
};

TEST_P(SubtractMultiplyToMultiplyAddTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<SubtractMultiplyToMultiplyAddTransformationTestValues> testValues = {
    // Multiply {} -> Multiply + Subtract {1x3x1x1}
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {0.1f}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{0.1f}, {ngraph::element::f32}},
            {}
        },
    },
    // Multiply {} -> Multiply + Subtract {1x3x1x1}
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {{0.1f, 0.2f, 0.3f}}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{0.1f, 0.2f, 0.3f}, {ngraph::element::f32}},
            {}
        },
    },
    // FP32 Subtract + Multiply {} -> Multiply + Subtract {1x3x1x1}
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{ngraph::element::f32}, {{128.f, 128.f / 2.f, 128.f / 4.f}}, {0.1f}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{0.1f}, {ngraph::element::f32}},
            {{-12.8f, -12.8f / 2.f, -12.8f / 4.f}, {ngraph::element::f32}}
        },
    },
    // FP32 Subtract + Multiply {} -> Multiply + Subtract {1x3x1x1}
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{ngraph::element::f32}, {128}, {0.1f}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{0.1f}, {ngraph::element::f32}},
            {{-12.8f}, {ngraph::element::f32}}
        },
    },
    // U8 Multiply {} -> Multiply + Subtract {1x3x1x1}
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}},
            ngraph::element::f32,
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{0.1f}, {ngraph::element::f32}},
            {}
        },
    },
    // U8 Subtract + Multiply {} -> Multiply + Subtract {1x3x1x1}
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}},
            ngraph::element::f32,
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{0.1f}, {ngraph::element::f32}},
            {{-12.8f}, {ngraph::element::f32}}
        },
    },
    // empty
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {}},
            ngraph::element::f32,
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {}},
            ngraph::element::u8,
            {},
            {}
        },
    },
    // empty
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {},
            {}
        },
    },
    // FP32 Multiply {5x1x1} -> Multiply + Subtract {1x5x1x1}
    {
        {2, 5, 4, 4},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, ngraph::element::f32, {5, 1, 1}}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, {ngraph::element::f32}, {5, 1, 1}},
            {}
        },
    },
    // FP32 Multiply {5x1x2}
    {
        {2, 5, 2, 2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f}, ngraph::element::f32, {5, 1, 2}}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {{}, {}, {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f}, ngraph::element::f32, {5, 1, 2}}},
            ngraph::element::f32,
            {},
            {}
        },
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    SubtractMultiplyToMultiplyAddTransformation,
    ::testing::ValuesIn(testValues),
    SubtractMultiplyToMultiplyAddTransformation::getTestCaseName);

} // namespace
