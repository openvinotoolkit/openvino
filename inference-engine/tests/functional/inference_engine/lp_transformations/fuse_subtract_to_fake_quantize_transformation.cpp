// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/fuse_subtract_to_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class FuseSubtractToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        FakeQuantizeOnData fakeQuantizeOnData;
        DequantizationOperations dequantization;
        FakeQuantizeOnData fakeQuantizeOnData2;
        DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        FakeQuantizeOnData fakeQuantizeOnData;
        DequantizationOperations dequantization;
        FakeQuantizeOnData fakeQuantizeOnData2;
        DequantizationOperations dequantization2;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class FuseSubtractToFakeQuantizeTransformation : public LayerTransformation,
    public testing::WithParamInterface<FuseSubtractToFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseSubtractToFakeQuantizeTransformationTestValues testValues = GetParam();

        actualFunction = testValues.actual.fakeQuantizeOnData2.empty() ?
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                testValues.inputShape,
                testValues.actual.fakeQuantizeOnData,
                testValues.actual.dequantization) :
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                testValues.inputShape,
                testValues.actual.fakeQuantizeOnData,
                testValues.actual.dequantization,
                testValues.actual.fakeQuantizeOnData2,
                testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation, ngraph::opset1::Subtract>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = testValues.expected.fakeQuantizeOnData2.empty() ?
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                testValues.inputShape,
                testValues.expected.fakeQuantizeOnData,
                testValues.expected.dequantization) :
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                testValues.inputShape,
                testValues.expected.fakeQuantizeOnData,
                testValues.expected.dequantization,
                testValues.expected.fakeQuantizeOnData2,
                testValues.expected.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationTestValues> obj) {
        const FuseSubtractToFakeQuantizeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.params.updatePrecisions << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData2 << "_" <<
            testValues.expected.dequantization2;
        return result.str();
    }
};

TEST_P(FuseSubtractToFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<FuseSubtractToFakeQuantizeTransformationTestValues> testValues = {
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::i8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::i8 },
            { {}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        Shape{1, 4, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    FuseSubtractToFakeQuantizeTransformation,
    ::testing::ValuesIn(testValues),
    FuseSubtractToFakeQuantizeTransformation::getTestCaseName);

} // namespace
