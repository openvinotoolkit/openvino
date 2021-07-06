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
#include <low_precision/network_helper.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/move_dequantization_after_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class MoveDequantizationAfterTransformationParams {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::element::Type originalPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool updatePrecision;
    bool moveSubtract;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::Shape,
    MoveDequantizationAfterTransformationParams> MoveDequantizationAfterTransformationTestValues;

class MoveDequantizationAfterTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<MoveDequantizationAfterTransformationTestValues> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());
        actualFunction = ngraph::builder::subgraph::MoveDequantizationAfterFunction::getOriginal(
            testValues.originalPrecision,
            inputShape,
            testValues.actual.dequantization);

        const auto targetNode = actualFunction->get_output_op(0)->get_input_node_shared_ptr(0);
        const auto dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(targetNode);
        ngraph::pass::low_precision::NetworkHelper::moveDequantizationAfter(
            targetNode,
            dequantization,
            testValues.updatePrecision,
            testValues.moveSubtract);

        referenceFunction = ngraph::builder::subgraph::MoveDequantizationAfterFunction::getReference(
            testValues.originalPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MoveDequantizationAfterTransformationTestValues> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            testValues.originalPrecision << "_" <<
            inputShape << "_" <<
            testValues.actual.dequantization << "_" <<
            (testValues.moveSubtract ? "move_subtract_" : "don't_move_subtract_") <<
            (testValues.updatePrecision ? "updatePrecision" : "don't_update_precision");
        return result.str();
    }
};

TEST_P(MoveDequantizationAfterTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, false, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> inputShapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

const std::vector<MoveDequantizationAfterTransformationParams> testValues = {
    // U8
    {
        ngraph::element::u8,
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false
    {
        ngraph::element::u8,
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ngraph::element::f32},  { { 7.f }, ngraph::element::f32, {}, false }, {} },
            ngraph::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // updatePrecision = false
    {
        ngraph::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        true,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ngraph::element::f32,
            { {},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false & updatePrecision = false
    {
        ngraph::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ngraph::element::f32},  { { 7.f }, ngraph::element::f32, {}, false }, {} },
            ngraph::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // I8
    {
        ngraph::element::i8,
        LayerTransformation::createParamsI8I8(),
        true,
        true,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ngraph::element::i8,
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false
    {
        ngraph::element::i8,
        LayerTransformation::createParamsI8I8(),
        true,
        false,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ngraph::element::f32},  { { 7.f }, ngraph::element::f32, {}, false }, {} },
            ngraph::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // updatePrecision = false
    {
        ngraph::element::i8,
        LayerTransformation::createParamsI8I8(),
        false,
        true,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ngraph::element::f32,
            { {},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false & updatePrecision = false
    {
        ngraph::element::i8,
        LayerTransformation::createParamsI8I8(),
        false,
        false,
        {
            { {ngraph::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ngraph::element::f32},  { { 7.f }, ngraph::element::f32, {}, false }, {} },
            ngraph::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // per-channel quantizations with the same values
    {
        ngraph::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            { {ngraph::element::f32},  { { 7.f, 7.f, 7.f } }, { { 10.f, 10.f, 10.f } } },
        },
        {
            { {ngraph::element::f32},  { { 7.f, 7.f, 7.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false }, {} },
            ngraph::element::f32,
            { {},  {}, { { 10.f, 10.f, 10.f } } },
        },
    },
    // per-channel quantizations with different values
    {
        ngraph::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            { {ngraph::element::f32},  { { 7.f, 8.f, 9.f } }, { { 10.f, 12.f, 16.f } } },
        },
        {
            { {ngraph::element::f32},  { { 7.f, 8.f, 9.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false }, {} },
            ngraph::element::f32,
            { {},  {}, { { 10.f, 12.f, 16.f } } },
        },
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MoveDequantizationAfterTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    MoveDequantizationAfterTransformation::getTestCaseName);
