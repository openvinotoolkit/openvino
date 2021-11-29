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

bool ngraph::op::is_unary_elementwise_arithmetic(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::UnaryElementwiseArithmetic*>(node) != nullptr;
}

bool ngraph::op::is_binary_elementwise_arithmetic(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::BinaryElementwiseArithmetic*>(node) != nullptr;
}

bool ngraph::op::is_binary_elementwise_comparison(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::BinaryElementwiseComparison*>(node) != nullptr;
}

bool ngraph::op::is_binary_elementwise_logical(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::BinaryElementwiseLogical*>(node) != nullptr;
}

bool ngraph::op::supports_auto_broadcast(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::v1::Select*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v0::SquaredDifference*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::util::BinaryElementwiseComparison*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::util::BinaryElementwiseLogical*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::util::BinaryElementwiseArithmetic*>(node) != nullptr;
}

bool ngraph::op::supports_decompose(const ngraph::Node* node)
{
    NGRAPH_SUPPRESS_DEPRECATED_START
    return dynamic_cast<const ngraph::op::util::FusedOp*>(node) != nullptr;
    NGRAPH_SUPPRESS_DEPRECATED_END
}

bool ngraph::op::is_op(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Op*>(node) != nullptr;
}

bool ngraph::op::is_parameter(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Parameter*>(node) != nullptr;
}

bool ngraph::op::is_output(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Result*>(node) != nullptr;
}

bool ngraph::op::is_sink(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Sink*>(node) != nullptr;
}

bool ngraph::op::is_constant(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Constant*>(node) != nullptr;
}

bool ngraph::op::is_commutative(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::v1::Add*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::Maximum*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::Equal*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::NotEqual*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::LogicalAnd*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v0::Xor*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::LogicalXor*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::Minimum*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::Multiply*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::v1::LogicalOr*>(node) != nullptr;
}

bool ngraph::op::is_unary_elementwise_arithmetic(const std::shared_ptr<ngraph::Node>& node)
{
    return is_unary_elementwise_arithmetic(node.get());
}
bool ngraph::op::is_binary_elementwise_arithmetic(const std::shared_ptr<ngraph::Node>& node)
{
    return is_binary_elementwise_arithmetic(node.get());
}
bool ngraph::op::is_binary_elementwise_comparison(const std::shared_ptr<ngraph::Node>& node)
{
    return is_binary_elementwise_comparison(node.get());
}
bool ngraph::op::is_binary_elementwise_logical(const std::shared_ptr<ngraph::Node>& node)
{
    return is_binary_elementwise_logical(node.get());
}

bool ngraph::op::supports_auto_broadcast(const std::shared_ptr<ngraph::Node>& node)
{
    return supports_auto_broadcast(node.get());
}

bool ngraph::op::supports_decompose(const std::shared_ptr<ngraph::Node>& node)
{
    return supports_decompose(node.get());
}

bool ngraph::op::is_op(const std::shared_ptr<ngraph::Node>& node)
{
    return is_op(node.get());
}
bool ngraph::op::is_parameter(const std::shared_ptr<ngraph::Node>& node)
{
    return is_parameter(node.get());
}
bool ngraph::op::is_output(const std::shared_ptr<ngraph::Node>& node)
{
    return is_output(node.get());
}
bool ngraph::op::is_sink(const std::shared_ptr<ngraph::Node>& node)
{
    return is_sink(node.get());
}
bool ngraph::op::is_constant(const std::shared_ptr<ngraph::Node>& node)
{
    return is_constant(node.get());
}
bool ngraph::op::is_commutative(const std::shared_ptr<ngraph::Node>& node)
{
    return is_commutative(node.get());
}
