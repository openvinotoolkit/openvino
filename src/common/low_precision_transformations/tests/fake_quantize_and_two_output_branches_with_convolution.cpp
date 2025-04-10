// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"

#include "common_test_utils/ov_test_utils.hpp"

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/fake_quantize_and_two_output_branches_with_convolution.hpp"

#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues {
public:
    class ActualValues {
    public:
        ov::builder::subgraph::FakeQuantizeOnData fqOnData;
        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
    };

    class ExpectedValues {
    public:
        ov::builder::subgraph::FakeQuantizeOnData fqOnData;
        ov::element::Type precisionBeforeOp;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOp;
        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights1;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter1;
        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights2;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter2;
    };

    TestTransformationParams params;
    ActualValues actual;
    ExpectedValues expected;
};

typedef std::tuple<
    ov::element::Type,
    ov::Shape,
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues> FakeQuantizeAndTwoOutputBranchesWithConvolutionParams;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::Shape shape = std::get<1>(GetParam());
        const FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues testValues = std::get<2>(GetParam());

        actualFunction = ov::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
            precision,
            shape,
            testValues.actual.fqOnData,
            testValues.actual.fqOnWeights1,
            testValues.actual.fqOnWeights2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getReference(
            precision,
            shape,
            TestTransformationParams::toParams(testValues.params),
            testValues.expected.fqOnData,
            testValues.expected.precisionBeforeOp,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOp,
            testValues.expected.dequantizationAfter1,
            testValues.expected.dequantizationAfter2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const ov::Shape shape = std::get<1>(obj.param);
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
    auto res = compare_functions(actualFunction, referenceFunction, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
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
            ov::element::u8,
            {{}, {}, {}},
            ov::element::f32,
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {{}, {}, {{ 1.f }, ov::element::f32, {}}},
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {{}, {}, {{ 1.f }, ov::element::f32, {}}},
        }
    },
    // TODO: LPT: issue #58685
//    // not update precisions
//    {
//        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
//        {
//            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
//            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
//            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
//        },
//        {
//            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
//            ov::element::f32,
//            {{}, {}, {}},
//            ov::element::f32,
//            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
//            {{}, {}, {{ 1.f }, ov::element::f32, { 1, 1, 1, 1 }}},
//            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
//            {{}, {}, {{ 1.f }, ov::element::f32, { 1, 1, 1, 1 }}},
//        }
//    },
    // not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
            ov::element::f32,
            {{}, {}, {}},
            ov::element::f32,
            { },
            {{}, {}, {{ 1.f }, ov::element::f32, {}}},
            { },
            {{}, {}, {{ 1.f }, ov::element::f32, {}}},
        }
    }
};

const std::vector<ov::Shape> shapes = {
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
