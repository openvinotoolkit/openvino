// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "ngraph/node.hpp"
#include "null_node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        constexpr NodeTypeInfo NullNode::type_info;

        std::shared_ptr<ngraph::Node>
            NullNode::clone_with_new_inputs(const OutputVector& /* new_args */) const
        {
            return std::make_shared<NullNode>();
        }
    } // namespace onnx_import
} // namespace ngraph

bool ngraph::op::is_null(const ngraph::Node* node)
{
    return dynamic_cast<const ngraph::onnx_import::NullNode*>(node) != nullptr;
}

bool ngraph::op::is_null(const std::shared_ptr<ngraph::Node>& node)
{
    return is_null(node.get());
}

bool ngraph::op::is_null(const Output<ngraph::Node>& output)
{
    return is_null(output.get_node());
}
