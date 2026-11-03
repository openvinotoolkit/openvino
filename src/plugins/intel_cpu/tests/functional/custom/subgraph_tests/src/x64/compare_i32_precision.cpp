// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

namespace {

enum class CompareType { EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL };

using CompareI32Params = std::tuple<CompareType, bool>;  // comparison, followed by LogicalNot

bool reference(CompareType type, int32_t lhs, int32_t rhs) {
    switch (type) {
    case CompareType::EQUAL:
        return lhs == rhs;
    case CompareType::NOT_EQUAL:
        return lhs != rhs;
    case CompareType::GREATER:
        return lhs > rhs;
    case CompareType::GREATER_EQUAL:
        return lhs >= rhs;
    case CompareType::LESS:
        return lhs < rhs;
    default:
        return lhs <= rhs;
    }
}

std::shared_ptr<ov::Node> make_compare(CompareType type,
                                       const ov::Output<ov::Node>& lhs,
                                       const ov::Output<ov::Node>& rhs) {
    switch (type) {
    case CompareType::EQUAL:
        return std::make_shared<ov::op::v1::Equal>(lhs, rhs);
    case CompareType::NOT_EQUAL:
        return std::make_shared<ov::op::v1::NotEqual>(lhs, rhs);
    case CompareType::GREATER:
        return std::make_shared<ov::op::v1::Greater>(lhs, rhs);
    case CompareType::GREATER_EQUAL:
        return std::make_shared<ov::op::v1::GreaterEqual>(lhs, rhs);
    case CompareType::LESS:
        return std::make_shared<ov::op::v1::Less>(lhs, rhs);
    default:
        return std::make_shared<ov::op::v1::LessEqual>(lhs, rhs);
    }
}

// Comparisons of i32 inputs must be exact. Values above 2^24 do not fit in f32, so comparing them as f32 gives wrong
// results (ticket 39337).
class CompareI32PrecisionTest : public testing::TestWithParam<CompareI32Params> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompareI32Params>& obj) {
        const auto& [type, negate] = obj.param;
        return "type=" + std::to_string(static_cast<int>(type)) + "_negate=" + std::to_string(negate);
    }
};

TEST_P(CompareI32PrecisionTest, ResultsAreExact) {
    const auto& [type, negate] = GetParam();
    constexpr int32_t max = std::numeric_limits<int32_t>::max();
    constexpr int32_t min = std::numeric_limits<int32_t>::min();
    const std::vector<std::pair<int32_t, int32_t>> pairs = {{16777217, 16777216},
                                                            {16777216, 16777217},
                                                            {16777217, 16777217},
                                                            {-16777217, -16777216},
                                                            {-16777216, -16777217},
                                                            {max, max - 1},
                                                            {min, min + 1},
                                                            {max, min},
                                                            {min, max},
                                                            {0, 0},
                                                            {1, -1},
                                                            {-1, 1}};

    // 29 elements cover full vectors of every ISA and the scalar tail.
    constexpr size_t size = 29;
    const ov::Shape shape{1, size};
    std::vector<int32_t> lhs_values(size);
    std::vector<int32_t> rhs_values(size);
    for (size_t i = 0; i < size; ++i) {
        lhs_values[i] = pairs[i % pairs.size()].first;
        rhs_values[i] = pairs[i % pairs.size()].second;
    }

    auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape);
    auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape);
    std::shared_ptr<ov::Node> output = make_compare(type, lhs, rhs);
    if (negate) {
        output = std::make_shared<ov::op::v1::LogicalNot>(output);
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{lhs, rhs});

    ov::Core core;
    auto request = core.compile_model(model, "CPU").create_infer_request();
    request.set_input_tensor(0, ov::Tensor(ov::element::i32, shape, lhs_values.data()));
    request.set_input_tensor(1, ov::Tensor(ov::element::i32, shape, rhs_values.data()));
    request.infer();

    const auto result = request.get_output_tensor(0);
    ASSERT_EQ(result.get_element_type(), ov::element::boolean);
    const auto* actual = result.data<const char>();
    for (size_t i = 0; i < size; ++i) {
        const bool expected = reference(type, lhs_values[i], rhs_values[i]) != negate;
        EXPECT_EQ(actual[i] != 0, expected) << "at index " << i << ": " << lhs_values[i] << " vs " << rhs_values[i];
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_CompareI32Precision,
                         CompareI32PrecisionTest,
                         testing::Combine(testing::Values(CompareType::EQUAL,
                                                          CompareType::NOT_EQUAL,
                                                          CompareType::GREATER,
                                                          CompareType::GREATER_EQUAL,
                                                          CompareType::LESS,
                                                          CompareType::LESS_EQUAL),
                                          testing::Bool()),
                         CompareI32PrecisionTest::getTestCaseName);

}  // namespace
