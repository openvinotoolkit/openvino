// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/squeeze.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/network_helper.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/standardize_dequantization_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class StandardizeDequantizationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::Shape inputShape;
    Actual actual;
    Expected expected;
};

class StandardizeDequantization : public LayerTransformation, public testing::WithParamInterface<StandardizeDequantizationTestValues> {
public:
    void SetUp() override {
        const StandardizeDequantizationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::StandardizeDequantizationFunction::getOriginal(
            testValues.actual.precisionBeforeDequantization,
            testValues.inputShape,
            testValues.actual.dequantization);
        const auto targetNode = actualFunction->get_output_op(0)->get_input_node_shared_ptr(0);
        auto dequantization = low_precision::NetworkHelper::getDequantization(targetNode);
        low_precision::NetworkHelper::standardizeDequantization(dequantization);
        referenceFunction = ngraph::builder::subgraph::StandardizeDequantizationFunction::getOriginal(
            testValues.expected.precisionBeforeDequantization,
            testValues.inputShape,
            testValues.expected.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<StandardizeDequantizationTestValues> obj) {
        const StandardizeDequantizationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantization << "_";

        return result.str();
    }
};

TEST_P(StandardizeDequantization, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, false, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<StandardizeDequantizationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        {
            ngraph::element::f32,
            {
                {},
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 0 },
                { {10.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 0 }
            },
        },
        {
            ngraph::element::f32,
            {
                {},
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ngraph::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        {
            ngraph::element::i32,
            {
                {ngraph::element::f32},
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 },
                { {10.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 0 }
            },
        },
        {
            ngraph::element::i32,
            {
                { ngraph::element::f32 },
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ngraph::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        {
            ngraph::element::u32,
            {
                { ngraph::element::f32 },
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 0 },
                { {10.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 }
            },
        },
        {
            ngraph::element::u32,
            {
                { {ngraph::element::f32} },
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ngraph::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(true),
        { 1, 3, 16, 16 },
        {
            ngraph::element::u32,
            {
                { ngraph::element::f32 },
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 },
                { {10.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 }
            },
        },
        {
            ngraph::element::u32,
            {
                { ngraph::element::f32 },
                { {7.f}, ngraph::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ngraph::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    StandardizeDequantization,
        ::testing::ValuesIn(testValues),
    StandardizeDequantization::getTestCaseName);