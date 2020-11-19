// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution_function.hpp"

#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues {
public:
    low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ActualValues actual;
    ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ExpectedValues expected;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues& testValue) {
    return out << "_" <<
        testValue.params.precisionsOnActivations[0] << "_" <<
        testValue.actual.fqOnData << "_" <<
        testValue.actual.fqOnWeights1 << "_" <<
        testValue.actual.fqOnWeights2;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues> FakeQuantizeAndTwoOutputBranchesWithConvolutionParams;

class FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool updatePrecision = std::get<2>(GetParam());
        const FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues testValues = std::get<3>(GetParam());

        const low_precision::LayerTransformation::Params params = low_precision::LayerTransformation::Params(testValues.params).
            setUpdatePrecisions(updatePrecision);

        actualFunction = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
            precision,
            shape,
            testValues.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(params);
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getReference(
            precision,
            shape,
            params,
            testValues.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const bool updatePrecision = std::get<2>(obj.param);
        const FakeQuantizeAndTwoOutputBranchesWithConvolutionTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) <<
            (updatePrecision ? "" : "_notUpdatePrecision_") <<
            testValues;
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

const std::vector<bool> updatePrecisions = {
    true,
    false
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
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            { 1.f },
            { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            { 1.f }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(fakeQuantizeOnDataTestValues)),
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName);
