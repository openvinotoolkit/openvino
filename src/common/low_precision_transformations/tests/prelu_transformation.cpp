// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/prelu.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/prelu.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ov::pass;
using namespace ov;

class PReluTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ov::PartialShape, PReluTransformationTestValues> PReluTransformationParams;

class PReluTransformation : public LayerTransformation, public testing::WithParamInterface<PReluTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());

        actualFunction =
            ov::builder::subgraph::PReluFunction::getOriginal(inputShape,
                                                                  testValues.actual.precisionBeforeDequantization,
                                                                  testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::PReluTransformation, ov::op::v0::PRelu>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            ov::builder::subgraph::PReluFunction::getReference(inputShape,
                                                                   testValues.expected.precisionBeforeDequantization,
                                                                   testValues.expected.dequantizationBefore,
                                                                   testValues.expected.precisionAfterOperation,
                                                                   testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<PReluTransformationParams> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << inputShape << "_" << testValues.actual.precisionBeforeDequantization << "_"
               << testValues.actual.dequantization << "_" << testValues.expected.dequantizationBefore;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> actualFunction;
    std::shared_ptr<ov::Model> referenceFunction;
};

TEST_P(PReluTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> shapes = {{1, 3, 16, 16}, {-1, -1, -1, -1}, {1, 1, 2, 3, 4, 16}, {5}};

const std::vector<PReluTransformationTestValues> testValues = {
    // U8: no subtract
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {0.1f}}},
     {ov::element::u8, {{}, {}, {}}, ov::element::f32, {{}, {}, {0.1f}}}},
    // I8: no subtract
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {}, {0.1f}}},
     {ov::element::i8, {{}, {}, {}}, ov::element::f32, {{}, {}, {0.1f}}}},
    // U8: with positive subtract value
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}},
     {ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}, ov::element::f32, {{}, {}, {}}}},
    // I8: with positive subtract value
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {127}, {0.1f}}},
     {ov::element::i8, {{ov::element::f32}, {127}, {0.1f}}, ov::element::f32, {{}, {}, {}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         PReluTransformation,
                         ::testing::Combine(::testing::ValuesIn(shapes), ::testing::ValuesIn(testValues)),
                         PReluTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> shapesWithDynamicRank = {PartialShape::dynamic()};

const std::vector<PReluTransformationTestValues> testValues = {
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {0.1f}}},
     {ov::element::u8, {{ov::element::f32}, {}, {0.1f}}, ov::element::f32, {{}, {}, {}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         PReluTransformation,
                         ::testing::Combine(::testing::ValuesIn(shapesWithDynamicRank),
                                            ::testing::ValuesIn(testValues)),
                         PReluTransformation::getTestCaseName);
}  // namespace testValues2

}  // namespace
