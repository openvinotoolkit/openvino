// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "low_precision/eliminate_fake_quantize.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/max_pool.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/fuse_fake_quantize.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ov;
using namespace ov::pass;

class TransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBefore;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData1;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData2;
    };

    class Expected {
    public:
        ov::element::Type precisionBefore;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData1;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData2;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOperations2;
    };

    ov::PartialShape inputShape;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class EliminateFakeQuantizeTransformation : public LayerTransformation,
                                            public testing::WithParamInterface<TransformationTestValues> {
public:
    void SetUp() override {
        const TransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::get(testValues.inputShape,
                                                                                  testValues.actual.precisionBefore,
                                                                                  testValues.actual.fakeQuantizeOnData1,
                                                                                  testValues.actual.fakeQuantizeOnData2,
                                                                                  {});
        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
        transformer.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(testValues.params);
        transformer.add<ov::pass::low_precision::FakeQuantizeTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
        transformer.add<ov::pass::low_precision::EliminateFakeQuantizeTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            ngraph::builder::subgraph::FuseFakeQuantizeFunction::get(testValues.inputShape,
                                                                     testValues.expected.precisionBefore,
                                                                     testValues.expected.fakeQuantizeOnData1,
                                                                     testValues.expected.fakeQuantizeOnData2,
                                                                     testValues.expected.dequantizationOperations2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<TransformationTestValues> obj) {
        const TransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.inputShape << "_" << testValues.params.updatePrecisions << "_"
               << testValues.actual.precisionBefore << "_" << testValues.actual.fakeQuantizeOnData1 << "_"
               << testValues.actual.fakeQuantizeOnData2 << "_" << testValues.expected.precisionBefore << "_"
               << testValues.expected.fakeQuantizeOnData1 << "_" << testValues.expected.fakeQuantizeOnData2 << "_"
               << testValues.expected.dequantizationOperations2;
        return result.str();
    }
};

TEST_P(EliminateFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

// clang-format off
const std::vector<TransformationTestValues> testValues = {
    {
        {1, 3, 16, 16},
        TestTransformationParams(true, {ov::element::u8}, {ov::element::i8}),
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, element::u8},
            {},
            { ov::element::f32, {}, {{0.01f}, ov::element::f32, {}} }
        }
    },
    {
        {1, 3, 16, 16},
        TestTransformationParams(true, {ov::element::u8}, {ov::element::i8}),
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {256ul, {}, {0.f}, {2.549f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, element::u8},
            {},
            { ov::element::f32, {}, {{0.01f}, ov::element::f32, {}} }
        }
    },
    {
        {1, 3, 16, 16},
        TestTransformationParams(true, {ov::element::u8}, {ov::element::i8}),
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f / 2.f}}
        },
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, element::u8},
            {},
            { ov::element::f32, {}, {{0.005f}, ov::element::f32, {}} }
        }
    },
    {
        {1, 3, 16, 16},
        TestTransformationParams(true, {ov::element::u8}, {ov::element::i8}),
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {256ul, {}, {0.f}, {2.55f / 2.f}, {0.f}, {2.55f / 2.f}}
        },
        {
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, element::u8},
            {256ul, {}, {0.f}, {127.5f}, {0.f}, {255.f}, element::u8},
            { ov::element::f32, {}, {{0.005f}, ov::element::f32, {}} }
        }
    }
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         EliminateFakeQuantizeTransformation,
                         ::testing::ValuesIn(testValues),
                         EliminateFakeQuantizeTransformation::getTestCaseName);

}  // namespace
