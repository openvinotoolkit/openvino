// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "transformations/low_precision/subtract_multiply_to_multiply_add.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/subtract_multiply_to_multiply_add_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class SubtrcatMultiplyToMultiplyAddTransformationTestValues {
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

class SubtrcatMultiplyToMultiplyAddTransformation : public LayerTransformation, public testing::WithParamInterface<SubtrcatMultiplyToMultiplyAddTransformationTestValues> {
public:
    void SetUp() override {
        SubtrcatMultiplyToMultiplyAddTransformationTestValues testValues = GetParam();

        actualFunction = SubtractMultiplyToMultiplyAddFunction::getOriginal(
            testValues.shape,
            testValues.actual.precisionBefore,
            testValues.actual.dequantization,
            testValues.actual.precisionAfter);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::SubtrcatMultiplyToMultiplyAddTransformation, ngraph::opset1::Multiply>(
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

    static std::string getTestCaseName(testing::TestParamInfo<SubtrcatMultiplyToMultiplyAddTransformationTestValues> obj) {
        SubtrcatMultiplyToMultiplyAddTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.actual.precisionBefore << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.precisionAfter << "_" <<
            testValues.expected.precisionBefore << "_" <<
            testValues.expected.dequantization << "_" <<
            testValues.expected.precisionAfter;
        return result.str();
    }
};

TEST_P(SubtrcatMultiplyToMultiplyAddTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<SubtrcatMultiplyToMultiplyAddTransformationTestValues> testValues = {
    // U8
    {
        {1, 3, 299, 299},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{ngraph::element::f32}, {{128}}, {{0.1f}}},
            ngraph::element::f32,
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{{0.1f}}, {ngraph::element::f32}},
            {{{-12.8f}}, {ngraph::element::f32}}
        },
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    SubtrcatMultiplyToMultiplyAddTransformation,
    ::testing::ValuesIn(testValues),
    SubtrcatMultiplyToMultiplyAddTransformation::getTestCaseName);

} // namespace
