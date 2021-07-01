// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include <low_precision/add.hpp>
#include "lpt_ngraph_functions/elementwise_with_multi_parent_dequantization_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class ElementwiseWithMultiParentDequantizationTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precision1;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type precision2;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ngraph::element::Type precision1;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type precision2;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
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

class ElementwiseWithMultiParentDequantizationTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ElementwiseWithMultiParentDequantizationTransformationTestValues> {
public:
    void SetUp() override {
        const ElementwiseWithMultiParentDequantizationTransformationTestValues testValues = GetParam();

        actualFunction = ElementwiseWithMultiParentDequantizationFunction::get(
            testValues.precision,
            testValues.inputShape,
            testValues.params,
            testValues.actual.precision1,
            testValues.actual.dequantization1,
            testValues.actual.precision2,
            testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AddTransformation, ngraph::opset1::Add>(
            low_precision::LayerTransformation::Params(testValues.params));
        transform.transform(actualFunction);

        referenceFunction = ElementwiseWithMultiParentDequantizationFunction::get(
            testValues.precision,
            testValues.inputShape,
            testValues.params,
            testValues.expected.precision1,
            testValues.expected.dequantization1,
            testValues.expected.precision2,
            testValues.expected.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ElementwiseWithMultiParentDequantizationTransformationTestValues> obj) {
        const ElementwiseWithMultiParentDequantizationTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.precision << "_" <<
            testValues.inputShape << "_" <<
            testValues.actual.precision1 << "_" <<
            testValues.actual.dequantization1 << "_" <<
            testValues.actual.precision2 << "_" <<
            testValues.actual.dequantization2;
        return result.str();
    }
};

TEST_P(ElementwiseWithMultiParentDequantizationTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ElementwiseWithMultiParentDequantizationTransformationTestValues> addTransformationTestValues = {
    // U8
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }},
            ngraph::element::u8,
            {},
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }},
            ngraph::element::u8,
            {},
        }
    },
    // U8
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ElementwiseWithMultiParentDequantizationTransformation,
    ::testing::ValuesIn(addTransformationTestValues),
    ElementwiseWithMultiParentDequantizationTransformation::getTestCaseName);
