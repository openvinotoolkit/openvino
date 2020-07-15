//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/op/util/binary_elementwise_logical.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/type.hpp"

bool ngraph::op::util::is_unary_elementwise_arithmetic(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::UnaryElementwiseArithmetic*>(node) != nullptr;
}

bool ngraph::op::util::is_binary_elementwise_arithmetic(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::BinaryElementwiseArithmetic*>(node) != nullptr;
}

bool ngraph::op::util::is_binary_elementwise_comparison(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::BinaryElementwiseComparison*>(node) != nullptr;
}

bool ngraph::op::util::is_binary_elementwise_logical(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::BinaryElementwiseLogical*>(node) != nullptr;
}

bool ngraph::op::util::supports_auto_broadcast(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::v1::Select*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::util::BinaryElementwiseComparison*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::util::BinaryElementwiseLogical*>(node) != nullptr ||
           dynamic_cast<const ngraph::op::util::BinaryElementwiseArithmetic*>(node) != nullptr;
}

bool ngraph::op::util::supports_decompose(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::util::FusedOp*>(node) != nullptr;
}

bool ngraph::op::util::is_op(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Op*>(node) != nullptr;
}

bool ngraph::op::util::is_parameter(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::op::Parameter*>(node) != nullptr;
}
