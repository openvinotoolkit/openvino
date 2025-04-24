// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/comparison.hpp"

#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/not_equal.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_comparison(const ov::Output<Node>& in0,
                                          const ov::Output<Node>& in1,
                                          ov::test::utils::ComparisonTypes comparison_type) {
    switch (comparison_type) {
    case ov::test::utils::ComparisonTypes::EQUAL:
        return std::make_shared<ov::op::v1::Equal>(in0, in1);
    case ov::test::utils::ComparisonTypes::NOT_EQUAL:
        return std::make_shared<ov::op::v1::NotEqual>(in0, in1);
    case ov::test::utils::ComparisonTypes::GREATER:
        return std::make_shared<ov::op::v1::Greater>(in0, in1);
    case ov::test::utils::ComparisonTypes::GREATER_EQUAL:
        return std::make_shared<ov::op::v1::GreaterEqual>(in0, in1);
    case ov::test::utils::ComparisonTypes::IS_FINITE:
        return std::make_shared<ov::op::v10::IsFinite>(in0);
    case ov::test::utils::ComparisonTypes::IS_INF:
        return std::make_shared<ov::op::v10::IsInf>(in0);
    case ov::test::utils::ComparisonTypes::IS_NAN:
        return std::make_shared<ov::op::v10::IsNaN>(in0);
    case ov::test::utils::ComparisonTypes::LESS:
        return std::make_shared<ov::op::v1::Less>(in0, in1);
    case ov::test::utils::ComparisonTypes::LESS_EQUAL:
        return std::make_shared<ov::op::v1::LessEqual>(in0, in1);
    default: {
        throw std::runtime_error("Incorrect type of Comparison operation");
    }
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
