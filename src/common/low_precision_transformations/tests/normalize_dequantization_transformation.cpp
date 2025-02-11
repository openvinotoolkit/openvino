// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "low_precision/network_helper.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/normalize_dequantization.hpp"

using namespace testing;
using namespace ov::pass;

class NormalizeDequantizationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };
    TestTransformationParams params;
    ov::Shape inputShape;
    bool constantPath;
    Actual actual;
    Expected expected;
};

class NormalizeDequantizationTransformation : public LayerTransformation, public testing::WithParamInterface<NormalizeDequantizationTestValues> {
public:
    void SetUp() override {
        const NormalizeDequantizationTestValues testValues = GetParam();

        actualFunction = ov::builder::subgraph::NormalizeDequantizationFunction::getOriginal(
            testValues.actual.precisionBeforeDequantization,
            testValues.inputShape,
            testValues.actual.dequantization,
            testValues.constantPath);

        const auto targetNode = actualFunction->get_output_op(0)->get_input_node_shared_ptr(0);
        const auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(targetNode);
        ov::pass::low_precision::NetworkHelper::normalizeDequantization(dequantization);

        referenceFunction = ov::builder::subgraph::NormalizeDequantizationFunction::getOriginal(
            testValues.expected.precisionBeforeDequantization,
            testValues.inputShape,
            testValues.expected.dequantization,
            testValues.constantPath);
    }

    static std::string getTestCaseName(testing::TestParamInfo<NormalizeDequantizationTestValues> obj) {
        const NormalizeDequantizationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantization << "_";

        return result.str();
    }
};

TEST_P(NormalizeDequantizationTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false, true);
    ASSERT_TRUE(res.first) << res.second;
}

using Subtract = ov::builder::subgraph::DequantizationOperations::Subtract;
using Multiply = ov::builder::subgraph::DequantizationOperations::Multiply;

const std::vector<NormalizeDequantizationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        false,
        {
            ov::element::f32,
            {
                {},
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 0 },
                { {10.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 0 }
            },
        },
        {
            ov::element::f32,
            {
                {},
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1},
                {{10.0f}, ov::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        false,
        {
            ov::element::i32,
            {
                {ov::element::f32},
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                { {10.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 0 }
            },
        },
        {
            ov::element::i32,
            {
                { ov::element::f32 },
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ov::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        false,
        {
            ov::element::u32,
            {
                { ov::element::f32 },
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 0 },
                { {10.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 }
            },
        },
        {
            ov::element::u32,
            {
                { {ov::element::f32} },
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ov::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(true),
        { 1, 3, 16, 16 },
        false,
        {
            ov::element::u32,
            {
                { ov::element::f32 },
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                { {10.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 }
            },
        },
        {
            ov::element::u32,
            {
                { ov::element::f32 },
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                {{10.0f}, ov::element::f32, {1, 3, 16, 16}, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        true,
        {
            ov::element::f32,
            {
                {},
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                { {10.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 }
            },
        },
        {
            ov::element::f32,
            {
                {},
                { {7.f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 },
                { {10.0f}, ov::element::f32, { 1, 3, 16, 16 }, true, 1 }
            }
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        true,
        {
            ov::element::i8,
            {
                {ov::element::f32},
                Subtract({7.f}, ov::element::f32, { 1, 3, 16, 16 }).setConstantPrecision(ov::element::f16).setAddConvert(true),
                Multiply({10.f}, ov::element::f32, { 1, 3, 16, 16 })
            },
        },
        {
            ov::element::i8,
            {
                {ov::element::f32},
                Subtract({7.f}, ov::element::f32, { 1, 3, 16, 16 }).setConstantPrecision(ov::element::f16).setAddConvert(true),
                Multiply({10.f}, ov::element::f32, { 1, 3, 16, 16 })
            },
        },
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 16, 16 },
        true,
        {
            ov::element::f32,
            {
                {},
                Subtract({7.f}, ov::element::f32, { 1, 3, 16, 16 }).setConstantPrecision(ov::element::f16).setAddConvert(true),
                Multiply({10.f}, ov::element::f32, { 1, 3, 16, 16 })
            },
        },
        {
            ov::element::f32,
            {
                {},
                Subtract({7.f}, ov::element::f32, { 1, 3, 16, 16 }).setConstantPrecision(ov::element::f16).setAddConvert(true),
                Multiply({10.f}, ov::element::f32, { 1, 3, 16, 16 })
            },
        },
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    NormalizeDequantizationTransformation,
    ::testing::ValuesIn(testValues),
    NormalizeDequantizationTransformation::getTestCaseName);
