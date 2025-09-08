// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/logical.hpp"

#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_logical(const ov::Output<Node>& in0,
                                       const ov::Output<Node>& in1,
                                       ov::test::utils::LogicalTypes logical_type) {
    switch (logical_type) {
    case ov::test::utils::LogicalTypes::LOGICAL_AND:
        return std::make_shared<ov::op::v1::LogicalAnd>(in0, in1);
    case ov::test::utils::LogicalTypes::LOGICAL_OR:
        return std::make_shared<ov::op::v1::LogicalOr>(in0, in1);
    case ov::test::utils::LogicalTypes::LOGICAL_NOT:
        return std::make_shared<ov::op::v1::LogicalNot>(in0);
    case ov::test::utils::LogicalTypes::LOGICAL_XOR:
        return std::make_shared<ov::op::v1::LogicalXor>(in0, in1);
    default: {
        OPENVINO_THROW("Incorrect type of Logical operation");
    }
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
