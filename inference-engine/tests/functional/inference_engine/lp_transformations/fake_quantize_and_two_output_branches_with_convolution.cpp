// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_two_output_branches_with_convolution_function.hpp"

#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues {
public:
    class ActualValues {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
    };

    class ExpectedValues {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
        ngraph::element::Type precisionBeforeOp;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOp;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter1;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter2;
    };

    low_precision::LayerTransformation::Params params;
    ActualValues actual;
    ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues> FakeQuantizeAndTwoOutputBranchesWithConvolutionParams;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
            precision,
            shape,
            testValues.actual.fqOnData,
            testValues.actual.fqOnWeights1,
            testValues.actual.fqOnWeights2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getReference(
            precision,
            shape,
            testValues.params,
            testValues.expected.fqOnData,
            testValues.expected.precisionBeforeOp,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOp,
            testValues.expected.dequantizationAfter1,
            testValues.expected.dequantizationAfter2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_"
               << testValues.expected.fqOnData << "_" << testValues.expected.dequantizationAfter1 << "_"
               << testValues.expected.dequantizationAfter2;
        return result.str();
    }
};

TEST_P(FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues> fakeQuantizeOnDataTestValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {{}, {}, {{ 1.f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {{}, {}, {{ 1.f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
        }
    },
    // not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {{}, {}, {{ 1.f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {{}, {}, {{ 1.f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(fakeQuantizeOnDataTestValues)),
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName);
