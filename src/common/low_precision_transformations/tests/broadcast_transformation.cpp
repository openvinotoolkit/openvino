// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "low_precision/broadcast.hpp"
#include "ov_lpt_models/broadcast.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace ov::pass;
using namespace ov::builder::subgraph;
using namespace ov::opset1;
using namespace ov;

class BroadcastTransformationTestValues {
public:
    class Pattern {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Shape tagetShape;
    Shape axesMapping;
    Pattern actual;
    Pattern expected;
};

typedef std::tuple<
    ov::PartialShape,
    bool,
    BroadcastTransformationTestValues> BroadcastTransformationParams;

class BroadcastTransformation : public LayerTransformation, public testing::WithParamInterface<BroadcastTransformationParams> {
public:
    void SetUp() override {
        const auto [inputShape, v1, testValues] = GetParam();
        actualFunction = BroadcastFunction::get(
            v1,
            inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantizationBefore,
            testValues.tagetShape,
            testValues.axesMapping,
            testValues.actual.dequantizationAfter);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::BroadcastTransformation, ov::opset1::Broadcast>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = BroadcastFunction::get(
            v1,
            inputShape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.tagetShape,
            testValues.axesMapping,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<BroadcastTransformationParams> obj) {
        const auto [inputShape, v1, testValues] = obj.param;
        std::ostringstream result;
        result <<
            v1 << "_" <<
            inputShape << "_" <<
            testValues.tagetShape << "_" <<
            testValues.axesMapping << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantizationBefore << "_" <<
            testValues.actual.dequantizationAfter << "_" <<
            testValues.expected.precisionBeforeDequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.dequantizationAfter;
        return result.str();
    }
};

TEST_P(BroadcastTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace hw_broadcast {
const std::vector<ov::PartialShape> inputShapes = {
    { 1, 3, 1, 1 },
};

const std::vector<BroadcastTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 9, 9},
        { 0, 1, 2, 3 },
        {
            ov::element::u8,
            {{ov::element::f32}, {0.1f}, {0.2f}},
            {{}, {}, {}},
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            {{ov::element::f32}, {0.1f}, {0.2f}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 3, 9, 9 },
        { 0, 1, 2, 3 },
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{0.1f, 0.2f, 0.3f}},
                {{0.4f, 0.5f, 0.6f}}
            }
        },
        {
            ov::element::u8,
            { {}, {}, {}},
            {
                {ov::element::f32},
                {{0.1f, 0.2f, 0.3f}},
                {{0.4f, 0.5f, 0.6f}}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    BroadcastTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn({ true, false }),
        ::testing::ValuesIn(testValues)),
    BroadcastTransformation::getTestCaseName);
} // hw_broadcast

namespace chw_broadcast {
const std::vector<ov::PartialShape> inputShapes = {
    { 1, 1, 1, 1 }
};

const std::vector<BroadcastTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 9, 9, 9},
        { 0, 1, 2, 3 },
        {
            ov::element::u8,
            {{ov::element::f32}, {0.1f}, {0.2f}},
            {{}, {}, {}},
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            {{ov::element::f32}, {0.1f}, {0.2f}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 1, 9, 9, 9},
        // empty axis mapping => bcast with 2 inputs is created
        {},
        {
            ov::element::u8,
            {{ov::element::f32}, {0.1f}, {0.2f}},
            {{}, {}, {}},
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            {{ov::element::f32}, {0.1f}, {0.2f}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    BroadcastTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn({ true, false }),
        ::testing::ValuesIn(testValues)),
    BroadcastTransformation::getTestCaseName);
} // chw_broadcast

} // namespace
