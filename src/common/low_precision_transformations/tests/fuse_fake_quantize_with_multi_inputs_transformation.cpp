// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/fake_quantize.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/fuse_fake_quantize.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ov;
using namespace ov::pass;

class FuseFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        std::vector<ov::builder::subgraph::FuseFakeQuantizeFunction::Branch> branches;
        ov::element::Type precisionFakeQuantizeOnData;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    };

    class Expected {
    public:
        std::vector<ov::builder::subgraph::FuseFakeQuantizeFunction::Branch> branches;
        ov::element::Type precisionFakeQuantizeOnData;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    ov::Shape inputShape;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class FuseFakeQuantizeWithMultiInputsTransformation : public LayerTransformation, public testing::WithParamInterface<FuseFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseFakeQuantizeTransformationTestValues testValues = GetParam();

        actualFunction = ov::builder::subgraph::FuseFakeQuantizeFunction::get(
            testValues.inputShape,
            testValues.actual.branches,
            testValues.actual.precisionFakeQuantizeOnData,
            testValues.actual.fakeQuantizeOnData);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::FakeQuantizeTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::FuseFakeQuantizeFunction::get(
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
    auto res = compare_functions(actualFunction, referenceFunction, false, true);
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
