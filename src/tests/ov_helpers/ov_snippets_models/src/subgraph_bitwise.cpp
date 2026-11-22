// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_bitwise.hpp"

#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

BitwiseFunction::BitwiseFunction(const std::vector<PartialShape>& inputShapes, ov::element::Type_t precision)
    : SnippetsFunctionBase(inputShapes, precision) {
    OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> BitwiseFunction::initOriginal() const {
    auto p0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto p1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    auto bitwise_and = std::make_shared<ov::op::v13::BitwiseAnd>(p0, p1);
    auto bitwise_or = std::make_shared<ov::op::v13::BitwiseOr>(bitwise_and, p1);
    auto bitwise_xor = std::make_shared<ov::op::v13::BitwiseXor>(bitwise_or, p0);
    auto bitwise_not = std::make_shared<ov::op::v13::BitwiseNot>(bitwise_xor);
    return std::make_shared<ov::Model>(ov::OutputVector{bitwise_not}, ov::ParameterVector{p0, p1});
}

std::shared_ptr<ov::Model> BitwiseFunction::initReference() const {
    auto p0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto p1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    auto ip0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto ip1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    auto bitwise_and = std::make_shared<ov::op::v13::BitwiseAnd>(ip0, ip1);
    auto bitwise_or = std::make_shared<ov::op::v13::BitwiseOr>(bitwise_and, ip1);
    auto bitwise_xor = std::make_shared<ov::op::v13::BitwiseXor>(bitwise_or, ip0);
    auto bitwise_not = std::make_shared<ov::op::v13::BitwiseNot>(bitwise_xor);
    auto result = std::make_shared<ov::snippets::op::Result>(bitwise_not);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{ip0, ip1});
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{p0, p1}, body);
    return std::make_shared<ov::Model>(ov::OutputVector{subgraph}, ov::ParameterVector{p0, p1});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
