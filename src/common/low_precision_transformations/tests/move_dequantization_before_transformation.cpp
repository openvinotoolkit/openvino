// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/op/concat.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/move_dequantization_before.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::builder::subgraph;

namespace {

std::shared_ptr<ov::Node> getConcat(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Concat>(op)) {
            return op;
        }
    }
    return nullptr;
}

void moveDequantizationBeforeConcat(const std::shared_ptr<ov::Model>& model) {
    const auto concat = getConcat(model);
    OPENVINO_ASSERT(concat != nullptr, "Concat is not found");
    const auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantizationBelow(concat);
    ov::pass::low_precision::NetworkHelper::moveDequantizationBefore(concat, dequantization, true);
}

}  // namespace

class MoveDequantizationBeforeTransformationTestValues {
public:
    ov::PartialShape inputShape;
    std::int64_t axis;
    DequantizationOperations actual;
    // dequantization expected in front of each Concat input
    std::vector<DequantizationOperations> expected;
};

class MoveDequantizationBeforeTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<MoveDequantizationBeforeTransformationTestValues> {
public:
    void SetUp() override {
        const auto testValues = GetParam();
        actualFunction = MoveDequantizationBeforeFunction::getOriginal(ov::element::u8,
                                                                       testValues.inputShape,
                                                                       testValues.axis,
                                                                       testValues.actual);
        moveDequantizationBeforeConcat(actualFunction);

        referenceFunction = MoveDequantizationBeforeFunction::getReference(ov::element::u8,
                                                                           testValues.inputShape,
                                                                           testValues.axis,
                                                                           testValues.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MoveDequantizationBeforeTransformationTestValues> obj) {
        const auto testValues = obj.param;

        std::ostringstream result;
        result << testValues.inputShape << "_axis_" << testValues.axis << "_" << testValues.actual;
        return result.str();
    }
};

TEST_P(MoveDequantizationBeforeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace {
const std::vector<MoveDequantizationBeforeTransformationTestValues> testValues = {
    // per-channel constants of the Concat rank, split by the Concat axis
    {
        {1, 2, 4, 4},
        1,
        {{ov::element::f32},
         {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4, 1, 1}},
         {{0.1f, 0.2f, 0.3f, 0.4f}, ov::element::f32, {1, 4, 1, 1}}},
        {
            {{ov::element::f32},
             {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
             {{0.1f, 0.2f}, ov::element::f32, {1, 2, 1, 1}}},
            {{ov::element::f32},
             {{3.f, 4.f}, ov::element::f32, {1, 2, 1, 1}},
             {{0.3f, 0.4f}, ov::element::f32, {1, 2, 1, 1}}},
        },
    },
    // lower-rank constants broadcast against the Concat aligned to the trailing dims: the
    // constant's last dim is the Concat axis, so it is padded to the Concat rank and split
    {
        {1, 1, 2, 3},
        3,
        {{ov::element::f32},
         {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 6}},
         {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, ov::element::f32, {1, 6}}},
        {
            {{ov::element::f32},
             {{1.f, 2.f, 3.f}, ov::element::f32, {1, 1, 1, 3}},
             {{0.1f, 0.2f, 0.3f}, ov::element::f32, {1, 1, 1, 3}}},
            {{ov::element::f32},
             {{4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 1, 3}},
             {{0.4f, 0.5f, 0.6f}, ov::element::f32, {1, 1, 1, 3}}},
        },
    },
    // lower-rank constants broadcast along the Concat axis: no split
    {
        {1, 1, 2, 6},
        1,
        {{ov::element::f32},
         {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 6}},
         {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, ov::element::f32, {1, 6}}},
        {
            {{ov::element::f32},
             {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 6}},
             {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, ov::element::f32, {1, 6}}},
            {{ov::element::f32},
             {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 6}},
             {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, ov::element::f32, {1, 6}}},
        },
    },
    // dynamic-rank Concat with scalar constants: no split needed
    {
        ov::PartialShape::dynamic(),
        1,
        {{ov::element::f32}, {2.f}, {0.2f}},
        {
            {{ov::element::f32}, {2.f}, {0.2f}},
            {{ov::element::f32}, {2.f}, {0.2f}},
        },
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MoveDequantizationBeforeTransformation,
                         ::testing::ValuesIn(testValues),
                         MoveDequantizationBeforeTransformation::getTestCaseName);
}  // namespace

// With a dynamic rank the Concat axis can't be mapped onto a non-scalar constant, so whether it has
// to be split per input is unknown.
TEST(LPT, MoveDequantizationBeforeTransformationDynamicRankNegative) {
    const std::vector<DequantizationOperations> dequantizations = {
        {{ov::element::f32}, {2.f}, {{0.1f, 0.2f}, ov::element::f32, {2, 1}}},
        {{ov::element::f32}, {{1.f, 2.f}, ov::element::f32, {2, 1}}, {0.2f}},
    };
    for (const auto& dequantization : dequantizations) {
        const auto model = MoveDequantizationBeforeFunction::getOriginal(ov::element::u8,
                                                                         ov::PartialShape::dynamic(),
                                                                         1,
                                                                         dequantization);
        EXPECT_THROW(moveDequantizationBeforeConcat(model), ov::Exception) << dequantization;
    }
}
