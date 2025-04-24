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
#include "low_precision/network_helper.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/move_dequantization_after.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class MoveDequantizationAfterTransformationParams {
public:
    class Actual {
    public:
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };
    MoveDequantizationAfterTransformationParams(ov::element::Type originalPrecision,
                                                TestTransformationParams params,
                                                bool updateOutputPrecision,
                                                bool moveSubtract,
                                                Actual actual,
                                                Expected expected,
                                                bool typeRelaxed = true)
        : originalPrecision(originalPrecision),
          params(params),
          updateOutputPrecision(updateOutputPrecision),
          moveSubtract(moveSubtract),
          actual(std::move(actual)),
          expected(std::move(expected)),
          typeRelaxed(typeRelaxed) {}

    ov::element::Type originalPrecision;
    TestTransformationParams params;
    bool updateOutputPrecision;
    bool moveSubtract;
    Actual actual;
    Expected expected;
    bool typeRelaxed;
};

typedef std::tuple<
    ov::Shape,
    MoveDequantizationAfterTransformationParams> MoveDequantizationAfterTransformationTestValues;

class MoveDequantizationAfterTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<MoveDequantizationAfterTransformationTestValues> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());
        actualFunction = ov::builder::subgraph::MoveDequantizationAfterFunction::getOriginal(
            testValues.originalPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.typeRelaxed);

        const auto targetNode = actualFunction->get_output_op(0)->get_input_node_shared_ptr(0);
        const auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(targetNode);
        ov::pass::low_precision::NetworkHelper::moveDequantizationAfter(
            targetNode,
            dequantization,
            testValues.updateOutputPrecision,
            testValues.moveSubtract);

        referenceFunction = ov::builder::subgraph::MoveDequantizationAfterFunction::getReference(
            testValues.originalPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            testValues.typeRelaxed);
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
            (testValues.updateOutputPrecision ? "updateOutputPrecision_" : "don't_update_precision_") <<
            (testValues.typeRelaxed ? "typeRelaxed" : "not_typeRelaxed");
        return result.str();
    }
};

TEST_P(MoveDequantizationAfterTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace {
const std::vector<ov::Shape> inputShapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

const std::vector<MoveDequantizationAfterTransformationParams> testValues = {
    // U8
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ov::element::u8,
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ov::element::f32},  { { 7.f }, ov::element::f32, {}, false }, {} },
            ov::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // updateOutputPrecision = false
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        true,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ov::element::f32,
            { {},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false & updateOutputPrecision = false
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ov::element::f32},  { { 7.f }, ov::element::f32, {}, false }, {} },
            ov::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // I8
    {
        ov::element::i8,
        LayerTransformation::createParamsI8I8(),
        true,
        true,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ov::element::i8,
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false
    {
        ov::element::i8,
        LayerTransformation::createParamsI8I8(),
        true,
        false,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ov::element::f32},  { { 7.f }, ov::element::f32, {}, false }, {} },
            ov::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // updateOutputPrecision = false
    {
        ov::element::i8,
        LayerTransformation::createParamsI8I8(),
        false,
        true,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {},  {}, {} },
            ov::element::f32,
            { {},  { 7.f }, { 10.f } },
        },
    },
    // moveSubtract = false & updateOutputPrecision = false
    {
        ov::element::i8,
        LayerTransformation::createParamsI8I8(),
        false,
        false,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {ov::element::f32},  { { 7.f }, ov::element::f32, {}, false }, {} },
            ov::element::f32,
            { {},  {}, { 10.f } },
        },
    },
    // per-channel quantizations with the same values
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            { {ov::element::f32},  { { 7.f, 7.f, 7.f } }, { { 10.f, 10.f, 10.f } } },
        },
        {
            { {ov::element::f32},  { { 7.f, 7.f, 7.f }, ov::element::f32, { 1, 3, 1, 1 }, false }, {} },
            ov::element::f32,
            { {},  {}, { { 10.f, 10.f, 10.f } } },
        },
    },
    // per-channel quantizations with different values
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        false,
        false,
        {
            { {ov::element::f32},  { { 7.f, 8.f, 9.f } }, { { 10.f, 12.f, 16.f } } },
        },
        {
            { {ov::element::f32},  { { 7.f, 8.f, 9.f }, ov::element::f32, { 1, 3, 1, 1 }, false }, {} },
            ov::element::f32,
            { {},  {}, { { 10.f, 12.f, 16.f } } },
        },
    },
    // updateOutputPrecision = true & typeRelaxed = false
    {
        ov::element::u8,
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        {
            { {ov::element::f32},  { 7.f }, { 10.f } },
        },
        {
            { {}, {}, {} },
            ov::element::u8,
            { {ov::element::f32}, { 7.f }, { 10.f } },
        },
        false
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MoveDequantizationAfterTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    MoveDequantizationAfterTransformation::getTestCaseName);
}  // namespace

TEST(LPT, MoveDequantizationAfterTransformationNegative) {
    const bool typeRelaxed = false;
    const bool updateOutputPrecision = false;

    auto model = ov::builder::subgraph::MoveDequantizationAfterFunction::getOriginal(
        ov::element::u8,
        ov::Shape{1, 3, 16, 16},
        ov::builder::subgraph::DequantizationOperations{{ov::element::f32}, {7.f}, {10.f}},
        typeRelaxed);

    const auto targetNode = model->get_output_op(0)->get_input_node_shared_ptr(0);
    const auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(targetNode);

    // updateOutputPrecision is supported only for type relaxed nodes
    EXPECT_ANY_THROW(ov::pass::low_precision::NetworkHelper::moveDequantizationAfter(targetNode,
                                                                                     dequantization,
                                                                                     updateOutputPrecision,
                                                                                     true));
}
