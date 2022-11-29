// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/assign_and_read_value.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/assign_and_read_value_function.hpp"
#include "simple_low_precision_transformer.hpp"
#include "low_precision/layer_transformation.hpp"


namespace {
using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;

class AssignTransformationTestValues {
public:
    class Actual {
    public:
        std::vector<float> constantValue;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        std::vector<float> constantValue;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
    bool FQAfterReadValue;
};

typedef std::tuple <
    ngraph::PartialShape,          // input shape
    element::Type,                 // input precision
    element::Type,                 // precision before dequantization
    size_t,                        // opset version
    AssignTransformationTestValues // test values
> AssignTransformationParams;

class AssignTransformation : public LayerTransformation, public testing::WithParamInterface<AssignTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const element::Type precision = std::get<1>(GetParam());
        const element::Type precisionBeforeDequantization = std::get<2>(GetParam());
        const size_t opsetVersion = std::get<3>(GetParam());
        const AssignTransformationTestValues testValues = std::get<4>(GetParam());
        const std::vector<ngraph::element::Type> defaultPrecisions = low_precision::precision_set::int8_int16_int32_support;
        const auto params = TestTransformationParams(testValues.params)
            .setDefaultPrecisions(defaultPrecisions);

        actualFunction = ngraph::builder::subgraph::AssignAndReadValueFunction::getOriginal(
            inputShape,
            precision,
            precisionBeforeDequantization,
            opsetVersion,
            testValues.FQAfterReadValue,
            testValues.actual.constantValue,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer({}, {}, { ngraph::element::f32, defaultPrecisions });
        transformer.add<ngraph::pass::low_precision::AssignAndReadValueTransformation, ngraph::opset6::Assign>(actualFunction, params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::AssignAndReadValueFunction::getReference(
                inputShape,
                precision,
                precisionBeforeDequantization,
                opsetVersion,
                testValues.FQAfterReadValue,
                testValues.expected.constantValue,
                testValues.expected.dequantizationBefore,
                testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AssignTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const element::Type precision = std::get<1>(obj.param);
        const element::Type precisionBeforeDequantization = std::get<2>(obj.param);
        const size_t opsetVersion = std::get<3>(obj.param);
        const AssignTransformationTestValues testValues = std::get<4>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
               inputShape << "_" << precision << "_" <<
               opsetVersion << "_" << testValues.FQAfterReadValue << "_" <<
               precisionBeforeDequantization << "_" <<
               testValues.actual.constantValue << "_" <<
               testValues.actual.dequantization;
        return result.str();
    }
};

TEST_P(AssignTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes = {
    ngraph::PartialShape({ 1, 3, 224, 224 }),
};

const element::TypeVector precisions = {
    element::f16, element::f32
};

const element::TypeVector precisionsBeforeDequantizations = {
    element::i8, element::u8,
    element::i16, element::u16,
    element::i32, element::u32,
};

const std::vector<size_t> opsetVersions = {
    3,
    6
};

const std::vector<AssignTransformationTestValues> testValues = {
    // general case, no subtract, FQ after ReadValue
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            {0},
            {{ngraph::element::f32}, {}, {3.f}}
        },
        // ExpectedValues
        {
            {0},
            {{}, {}, {}},
            {{ngraph::element::f32}, {}, {3.f}}
        },
        true
    },
    // no FQ after ReadValue
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            {0},
            {{ngraph::element::f32}, {}, {3.f}}
        },
        // ExpectedValues
        {
            {0},
            {{}, {}, {}},
            {{ngraph::element::f32}, {}, {3.f}}
        },
        false
    },
    // non-zero constant
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            {5},
            {{ngraph::element::f32}, {}, {3.f}}
        },
        // ExpectedValues
        {
            {5},
            {{ngraph::element::f32}, {}, {3.f}},
            {}
        },
        false
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    AssignTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(precisionsBeforeDequantizations),
        ::testing::ValuesIn(opsetVersions),
        ::testing::ValuesIn(testValues)),
    AssignTransformation::getTestCaseName);
} // namespace testValues1
} // namespace
