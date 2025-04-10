// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/fuse_convert.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/fuse_convert.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class FuseConvertTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantization;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantization;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    };

    bool constInput;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::PartialShape,
    FuseConvertTransformationTestValues> FuseConvertTransformationParams;

class FuseConvertTransformation : public LayerTransformation, public testing::WithParamInterface<FuseConvertTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape inputShape = std::get<0>(GetParam());
        const FuseConvertTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::FuseConvertFunction::get(
                inputShape,
                testValues.actual.inputPrecision,
                testValues.actual.dequantization,
                testValues.actual.fakeQuantize,
                testValues.constInput);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::FuseConvertTransformation, ov::op::v0::Convert>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::FuseConvertFunction::get(
                inputShape,
                testValues.expected.inputPrecision,
                testValues.expected.dequantization,
                testValues.expected.fakeQuantize,
                testValues.constInput);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseConvertTransformationParams> obj) {
        const ov::PartialShape inputShape = std::get<0>(obj.param);
        const FuseConvertTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
               "IS_" << inputShape << "_" <<
               "AIP_" << testValues.actual.inputPrecision << "_" <<
               "ADEQ_" << testValues.actual.dequantization << "_" <<
               "AFQ_" << testValues.actual.fakeQuantize << "_" <<
               "EIP_" << testValues.expected.inputPrecision << "_" <<
               "EDEQ_" << testValues.expected.dequantization << "_" <<
               "EFQ_" << testValues.expected.fakeQuantize << "_" <<
               testValues.constInput;
        return result.str();
    }
};

TEST_P(FuseConvertTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes = {
    {1, 4, 16, 16},
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
    PartialShape::dynamic()
};

const std::vector<FuseConvertTransformationTestValues> testValues = {
    // fuse to subtract
    {
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                { ov::element::f32 },
                {1.f},
                {0.45f}
            },
            {}
        },
        {
            ov::element::u8,
            {
                {},
                DequantizationOperations::Subtract({1.f}, ov::element::f32).setConstantPrecision(ov::element::f32),
                {0.45f}
            },
            {}
        }
    },
    // fuse to multiply
    {
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                { ov::element::f32 },
                {},
                {0.45f}
            },
            {}
        },
        {
            ov::element::u8,
            {
                {},
                {},
                DequantizationOperations::Multiply({0.45f}, ov::element::f32).setConstantPrecision(ov::element::f32)
            },
            {}
        }
    },
    // Convert with unexpected precision
    {
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::f32,
            {{ ov::element::i32 }, {}, {3.f}},
            {}
        },
        {
            ov::element::f32,
            {{ ov::element::i32 }, {}, {3.f}},
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseConvertTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    FuseConvertTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> inputShapes = {
    {1, 4, 16, 16},
};

const std::vector<FuseConvertTransformationTestValues> testValuesWithConstant = {
    //  Constant
    //      |
    //  Convert Const Const Const Const
    //        \  \     |     /  /
    //         \  \    |    /  /
    //            FakeQuantize
    //
    {
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {}},
            { 256, {}, {0.f}, {0.1f}, {0.f}, {0.1f}, ov::element::f32}
        },
        {
            ov::element::f32,
            {},
            { 256, {}, {0.f}, {0.1f}, {0.f}, {0.1f}, ov::element::f32}
        }
    },
    // fuse to const
    {
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                { ov::element::f32 },
                {1.f},
                {0.45f}
            },
            {}
        },
        {
            ov::element::f32,
            {
                {},
                {1.f},
                {0.45f}
            },
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseConvertTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValuesWithConstant)),
    FuseConvertTransformation::getTestCaseName);
} // namespace testValues2
} // namespace
