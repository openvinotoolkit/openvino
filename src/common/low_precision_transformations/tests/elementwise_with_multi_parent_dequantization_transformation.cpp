// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include "low_precision/add.hpp"
#include "ov_lpt_models/elementwise_with_multi_parent_dequantization.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class ElementwiseWithMultiParentDequantizationTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precision1;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::element::Type precision2;
        ov::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ov::element::Type precision1;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::element::Type precision2;
        ov::builder::subgraph::DequantizationOperations dequantization2;
    };

    ov::element::Type precision;
    ov::Shape inputShape;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class ElementwiseWithMultiParentDequantizationTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ElementwiseWithMultiParentDequantizationTransformationTestValues> {
public:
    void SetUp() override {
        const ElementwiseWithMultiParentDequantizationTransformationTestValues testValues = GetParam();

        actualFunction = ElementwiseWithMultiParentDequantizationFunction::get(
            testValues.precision,
            testValues.inputShape,
            TestTransformationParams::toParams(testValues.params),
            testValues.actual.precision1,
            testValues.actual.dequantization1,
            testValues.actual.precision2,
            testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::AddTransformation, ov::op::v1::Add>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ElementwiseWithMultiParentDequantizationFunction::get(
            testValues.precision,
            testValues.inputShape,
            TestTransformationParams::toParams(testValues.params),
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ElementwiseWithMultiParentDequantizationTransformationTestValues> addTransformationTestValues = {
    // U8
    {
        ov::element::f32,
        ov::Shape{1, 4, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { {ov::element::f32},  { 7.f }, { 10.f }},
            ov::element::u8,
            {},
        },
        {
            ov::element::u8,
            { {ov::element::f32},  { 7.f }, { 10.f }},
            ov::element::u8,
            {},
        }
    },
    // U8
    {
        ov::element::f32,
        ov::Shape{1, 4, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { {ov::element::f32},  { 7.f }, { 10.f }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { {ov::element::f32},  { 7.f }, { 10.f }}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ElementwiseWithMultiParentDequantizationTransformation,
    ::testing::ValuesIn(addTransformationTestValues),
    ElementwiseWithMultiParentDequantizationTransformation::getTestCaseName);
