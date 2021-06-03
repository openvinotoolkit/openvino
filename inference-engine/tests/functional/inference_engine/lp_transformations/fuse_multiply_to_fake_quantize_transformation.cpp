// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/fuse_multiply_to_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FuseMultiplyToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<size_t, FuseMultiplyToFakeQuantizeTransformationTestValues> FuseMultiplyToFakeQuantizeTransformationTestParams;

class FuseMultiplyToFakeQuantizeTransformation : public LayerTransformation,
    public testing::WithParamInterface<FuseMultiplyToFakeQuantizeTransformationTestParams> {
public:
    void SetUp() override {
        const size_t quantizationLevel = std::get<0>(GetParam());
        FuseMultiplyToFakeQuantizeTransformationTestValues testValues = std::get<1>(GetParam());
        if (!testValues.actual.fakeQuantizeOnData.empty()) {
            testValues.actual.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData.empty()) {
            testValues.expected.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }

        actualFunction = ngraph::builder::subgraph::FuseMultiplyToFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.actual.fakeQuantizeOnData,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation, ngraph::opset1::Multiply>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FuseMultiplyToFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.expected.fakeQuantizeOnData,
            testValues.expected.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseMultiplyToFakeQuantizeTransformationTestParams> obj) {
        const size_t quantizationLevel = std::get<0>(obj.param);
        FuseMultiplyToFakeQuantizeTransformationTestValues testValues = std::get<1>(obj.param);
        if (!testValues.actual.fakeQuantizeOnData.empty()) {
            testValues.actual.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData.empty()) {
            testValues.expected.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }

        std::ostringstream result;
        result << testValues.params.updatePrecisions << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization;
        return result.str();
    }
};

TEST_P(FuseMultiplyToFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

std::vector<size_t> quantizationLevels = { 256ul, 128ul };

const std::vector<FuseMultiplyToFakeQuantizeTransformationTestValues> testValues = {
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f } },
            { {}, {}, {} },
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::i8 },
            { {element::f32}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f } },
            { {}, {}, {} },
        }
    },
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f } },
            { {}, {}, {} },
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    FuseMultiplyToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(quantizationLevels),
        ::testing::ValuesIn(testValues)),
    FuseMultiplyToFakeQuantizeTransformation::getTestCaseName);

} // namespace
