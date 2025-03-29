// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/low_precision.hpp"
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::builder::subgraph;

class FQDecompositionWithSharedConstants : public LayerTransformation, public WithParamInterface<bool> {
public:
    void SetUp() override {
        const bool addIntervalsAlignment = GetParam();
        const auto shape = ov::Shape{1, 3, 40, 40};
        const auto input_precision = ov::element::f32;

        {
            auto input = std::make_shared<ov::op::v0::Parameter>(input_precision, shape);
            auto shared_il = ov::op::v0::Constant::create(input_precision, {}, {0.f});
            auto shared_ih = ov::op::v0::Constant::create(input_precision, {}, {25.5f});
            auto shared_ol = ov::op::v0::Constant::create(input_precision, {}, {0.f});
            auto shared_oh = ov::op::v0::Constant::create(input_precision, {}, {25.5f});
            auto fq_before =
                std::make_shared<ov::op::v0::FakeQuantize>(input, shared_il, shared_ih, shared_ol, shared_oh, 256);
            auto fq_after =
                std::make_shared<ov::op::v0::FakeQuantize>(fq_before, shared_il, shared_ih, shared_ol, shared_oh, 256);
            auto relu = std::make_shared<ov::op::v0::Relu>(fq_after);
            if (addIntervalsAlignment) {
                addAttributes(
                    {fq_before, fq_after},
                    {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)});
                addAttributes({fq_after, relu}, {ov::QuantizationAlignmentAttribute(true)});
            }
            ResultVector results{std::make_shared<ov::op::v0::Result>(relu)};
            actualFunction = std::make_shared<Model>(results, ParameterVector{input}, "FakeQuantizeFunction");
        }

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(
            LayerTransformation::createParamsU8I8());
        transform.transform(actualFunction);

        {
            auto input = std::make_shared<ov::op::v0::Parameter>(input_precision, shape);
            auto fqStructure =
                FakeQuantizeOnData{256ul, Shape({}), {0.f}, {25.5f}, {0.f}, {255.f}, ov::element::u8};
            auto deqStructure = DequantizationOperations{{element::f32}, {}, {0.1f}};
            auto fq_before = makeFakeQuantizeTypeRelaxed(input, input_precision, fqStructure);
            auto dq_before = makeDequantization(fq_before, deqStructure);
            auto fq_after = makeFakeQuantizeTypeRelaxed(dq_before, input_precision, fqStructure);
            auto dq_after = makeDequantization(fq_after, deqStructure);
            auto relu = std::make_shared<ov::op::v0::Relu>(dq_after);
            ResultVector results{std::make_shared<ov::op::v0::Result>(relu)};
            referenceFunction = std::make_shared<Model>(results, ParameterVector{input}, "FakeQuantizeFunction");
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<bool> obj) {
        const bool addIntervalsAlignment = obj.param;
        return addIntervalsAlignment ? "with_IntervalsAlignment" : "without_IntervalsAlignment";
    }
};

TEST_P(FQDecompositionWithSharedConstants, FQDecompositionWithSharedConstants) {
    actualFunction->validate_nodes_and_infer_types();

    auto comparator = FunctionsComparator::no_default();
    comparator.enable(FunctionsComparator::CmpValues::NODES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::PRECISIONS);
    auto res = comparator.compare(actualFunction, referenceFunction);
    ASSERT_TRUE(res.valid) << res.message;

    // additional check: FQ constants after transformation mustn't be shared
    for (const auto& n : actualFunction->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(n)) {
            EXPECT_EQ(n->get_output_target_inputs(0).size(), 1);
        }
    }
}
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         FQDecompositionWithSharedConstants,
                         ::testing::ValuesIn(std::vector<bool>{false, true}),
                         FQDecompositionWithSharedConstants::getTestCaseName);
}  // namespace
