// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/fuse_fake_quantize.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/fuse_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FuseFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        std::vector<ngraph::builder::subgraph::FuseFakeQuantizeFunction::Branch> branches;
        ngraph::element::Type precisionFakeQuantizeOnData;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    };

    class Expected {
    public:
        std::vector<ngraph::builder::subgraph::FuseFakeQuantizeFunction::Branch> branches;
        ngraph::element::Type precisionFakeQuantizeOnData;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class FuseFakeQuantizeWithMultiInputsTransformation : public LayerTransformation, public testing::WithParamInterface<FuseFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseFakeQuantizeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.actual.branches,
            testValues.actual.precisionFakeQuantizeOnData,
            testValues.actual.fakeQuantizeOnData);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FuseFakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.expected.branches,
            testValues.expected.precisionFakeQuantizeOnData,
            testValues.expected.fakeQuantizeOnData);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationTestValues> obj) {
        const FuseFakeQuantizeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.params.updatePrecisions << "_" <<
            testValues.actual.branches[0].dequantization << "_" <<
            testValues.actual.branches[1].dequantization << "_" <<
            testValues.actual.precisionFakeQuantizeOnData << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization;
        return result.str();
        return result.str();
    }
};

TEST_P(FuseFakeQuantizeWithMultiInputsTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, false, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<FuseFakeQuantizeTransformationTestValues> testValues = {
    // Multiply
    {
        Shape{1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            {
                {
                    element::f32,
                    { {}, {}, { 0.01f } },
                    element::f32
                },
                {
                    element::f32,
                    { {}, {}, { 0.01f } },
                    element::f32
                }
            },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            {
                {
                    element::f32,
                    { {}, {}, { 0.01f } },
                    element::f32
                },
                {
                    element::f32,
                    { {}, {}, { 0.01f } },
                    element::f32
                }
            },
            element::f32,
            { 256ul, {}, { 0.f }, { 255.f }, { 0.f }, { 2.55f } }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseFakeQuantizeWithMultiInputsTransformation,
    ::testing::ValuesIn(testValues),
    FuseFakeQuantizeWithMultiInputsTransformation::getTestCaseName);

} // namespace
