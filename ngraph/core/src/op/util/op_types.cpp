// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/squared_difference.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/op/util/binary_elementwise_logical.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/type.hpp"

bool ov::op::is_unary_elementwise_arithmetic(const ov::Node* node)
{
    return dynamic_cast<const ov::op::util::UnaryElementwiseArithmetic*>(node) != nullptr;
}

bool ov::op::is_binary_elementwise_arithmetic(const ov::Node* node)
{
    return dynamic_cast<const ov::op::util::BinaryElementwiseArithmetic*>(node) != nullptr;
}

bool ov::op::is_binary_elementwise_comparison(const ov::Node* node)
{
    return dynamic_cast<const ov::op::util::BinaryElementwiseComparison*>(node) != nullptr;
}

bool ov::op::is_binary_elementwise_logical(const ov::Node* node)
{
    return dynamic_cast<const ov::op::util::BinaryElementwiseLogical*>(node) != nullptr;
}

bool ov::op::supports_auto_broadcast(const ov::Node* node)
{
    return dynamic_cast<const ov::op::v1::Select*>(node) != nullptr ||
           dynamic_cast<const ov::op::v0::SquaredDifference*>(node) != nullptr ||
           dynamic_cast<const ov::op::util::BinaryElementwiseComparison*>(node) != nullptr ||
           dynamic_cast<const ov::op::util::BinaryElementwiseLogical*>(node) != nullptr ||
           dynamic_cast<const ov::op::util::BinaryElementwiseArithmetic*>(node) != nullptr;
}

bool ov::op::supports_decompose(const ov::Node* node)
{
    NGRAPH_SUPPRESS_DEPRECATED_START
    return dynamic_cast<const ov::op::util::FusedOp*>(node) != nullptr;
    NGRAPH_SUPPRESS_DEPRECATED_END
}

bool ov::op::is_op(const ov::Node* node)
{
    return dynamic_cast<const ov::op::Op*>(node) != nullptr;
}

bool ov::op::is_parameter(const ov::Node* node)
{
    return dynamic_cast<const ov::op::Parameter*>(node) != nullptr;
}

bool ov::op::is_output(const ov::Node* node)
{
    return dynamic_cast<const ov::op::Result*>(node) != nullptr;
}

bool ov::op::is_sink(const ov::Node* node)
{
    return dynamic_cast<const ov::op::Sink*>(node) != nullptr;
}

bool ov::op::is_constant(const ov::Node* node)
{
    return dynamic_cast<const ov::op::Constant*>(node) != nullptr;
}

bool ov::op::is_commutative(const ov::Node* node)
{
    return dynamic_cast<const ov::op::v1::Add*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::Maximum*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::Equal*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::NotEqual*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::LogicalAnd*>(node) != nullptr ||
           dynamic_cast<const ov::op::v0::Xor*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::LogicalXor*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::Minimum*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::Multiply*>(node) != nullptr ||
           dynamic_cast<const ov::op::v1::LogicalOr*>(node) != nullptr;
}

bool ov::op::is_unary_elementwise_arithmetic(const std::shared_ptr<ov::Node>& node)
{
    return is_unary_elementwise_arithmetic(node.get());
}
bool ov::op::is_binary_elementwise_arithmetic(const std::shared_ptr<ov::Node>& node)
{
    return is_binary_elementwise_arithmetic(node.get());
}
bool ov::op::is_binary_elementwise_comparison(const std::shared_ptr<ov::Node>& node)
{
    return is_binary_elementwise_comparison(node.get());
}
bool ov::op::is_binary_elementwise_logical(const std::shared_ptr<ov::Node>& node)
{
    return is_binary_elementwise_logical(node.get());
}

bool ov::op::supports_auto_broadcast(const std::shared_ptr<ov::Node>& node)
{
    return supports_auto_broadcast(node.get());
}

bool ov::op::supports_decompose(const std::shared_ptr<ov::Node>& node)
{
    return supports_decompose(node.get());
}

bool ov::op::is_op(const std::shared_ptr<ov::Node>& node)
{
    return is_op(node.get());
}
bool ov::op::is_parameter(const std::shared_ptr<ov::Node>& node)
{
    return is_parameter(node.get());
}
bool ov::op::is_output(const std::shared_ptr<ov::Node>& node)
{
    return is_output(node.get());
}
bool ov::op::is_sink(const std::shared_ptr<ov::Node>& node)
{
    return is_sink(node.get());
}
bool ov::op::is_constant(const std::shared_ptr<ov::Node>& node)
{
    return is_constant(node.get());
}
bool ov::op::is_commutative(const std::shared_ptr<ov::Node>& node)
{
    return is_commutative(node.get());
}
