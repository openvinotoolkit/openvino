// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/max_pool.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/max_pool.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"


using namespace testing;
using namespace ov::pass;
using namespace ov;

class MaxPoolTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantization2;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::PartialShape,
    MaxPoolTransformationTestValues> MaxPoolTransformationParams;

class MaxPoolTransformation : public LayerTransformation, public testing::WithParamInterface<MaxPoolTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape shape = std::get<0>(GetParam());
        const MaxPoolTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::MaxPoolFunction::get(
            shape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization1,
            testValues.actual.preicsionAfterOperation,
            testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::MaxPoolFunction::get(
            shape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantization1,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MaxPoolTransformationParams> obj) {
        const ov::PartialShape shape = std::get<0>(obj.param);
        const MaxPoolTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << testValues.actual.precisionBeforeDequantization << "_"<<
            shape << "_" << toString(testValues.params) << "_" <<
            testValues.actual.dequantization1 << "_" <<
            testValues.actual.dequantization2 << "_" <<
            testValues.expected.dequantization1 << "_" <<
            testValues.expected.dequantization2 << "_";
        return result.str();
    }
};

TEST_P(MaxPoolTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 72, 48 },
    { 4, 3, 72, 48 },
    { -1, -1, -1, -1 },
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
    // Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { {}, {}, { {0.02f}, ov::element::f32, {}, true, 1, ov::element::f32 }},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { ov::element::f32, {}, { {0.02f}, ov::element::f32, {}, true, 1, ov::element::f32 }}
        }
    },
    // Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                {},
                { {128.f}, ov::element::f32, {}, true, 1, ov::element::f32 },
                { {0.02f}, ov::element::f32, {}, true, 1, ov::element::f32 }
            },
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                ov::element::f32,
                { {128.f}, ov::element::f32, {}, true, 1, ov::element::f32 },
                { {0.02f}, ov::element::f32, {}, true, 1, ov::element::f32 }
            }
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { ov::element::f32, { 128 }, { 0.02f }},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { ov::element::f32, { 128 }, { 0.02f }}
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { ov::element::f32, {}, { 0.02f }},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { ov::element::f32, {}, { 0.02f }}
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ov::element::f32,
            { ov::element::f32, { 128 }, { 0.02f }},
            ov::element::f32,
            {}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            { {}, { 128 }, { 0.02f }}
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ov::element::f32,
            { ov::element::f32, {}, { 0.02f }},
            ov::element::f32,
            {}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            { {}, {}, { 0.02f }}
        }
    },
    // per-channel dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { ov::element::f32, {{128.f, 64.f, 32.f}}, {{0.02f, 0.01f, 0.03f}}},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, { {0.02f, 0.01f, 0.03f} }}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> shapesWithDynamicChannels = {
    PartialShape::dynamic()
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
    // per-tensor dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { ov::element::f32, {128.f}, {0.01f}},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { ov::element::f32, {128.f}, {0.01f}}
        }
    },
    // per-channel dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            { ov::element::f32, {{128.f, 64.f, 32.f}}, {{0.02f, 0.01f, 0.03f}}},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, { {0.02f, 0.01f, 0.03f} }},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
} // namespace testValues2
