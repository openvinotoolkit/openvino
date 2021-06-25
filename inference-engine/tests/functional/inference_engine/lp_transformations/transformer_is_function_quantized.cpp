// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <low_precision/fake_quantize.hpp>
#include <low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/convolution_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class TestValues {
public:
    builder::subgraph::FakeQuantizeOnData fqOnData;
    builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
};

inline std::ostream& operator<<(std::ostream& out, const TestValues& testValue) {
    return out << "_" << testValue.fqOnData << "_" << testValue.fqOnWeights;
}

class TransformerIsFunctionQuantized : public LayerTransformation, public testing::WithParamInterface<TestValues> {
public:
    void SetUp() override {
        const TestValues testValues = GetParam();
        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::get(
            Shape({ 1, 3, 16, 16 }),
            element::f32,
            testValues.fqOnData,
            std::vector<float>({1.f}),
            testValues.fqOnWeights);
    }

    static std::string getTestCaseName(testing::TestParamInfo<TestValues> obj) {
        std::ostringstream result;
        result << obj.param;
        return result.str();
    }
};

TEST_P(TransformerIsFunctionQuantized, isFunctionQuantized) {
    actualFunction->validate_nodes_and_infer_types();
    const bool isFunctionQuantized = ngraph::pass::low_precision::LowPrecisionTransformer::isFunctionQuantized(actualFunction);

    const TestValues testValues = GetParam();
    const bool expected = !testValues.fqOnData.empty() || !testValues.fqOnWeights.empty();
    ASSERT_EQ(expected, isFunctionQuantized);
}

const std::vector<TestValues> testValues = {
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
    },
    {
        {},
        { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {},
    },
    { {}, {} }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    TransformerIsFunctionQuantized,
    ::testing::ValuesIn(testValues),
    TransformerIsFunctionQuantized::getTestCaseName);

} // namespace
