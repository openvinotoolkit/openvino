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

#include <string>

#include "ngraph/node.hpp"
#include "null_node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        constexpr NodeTypeInfo NullNode::type_info;

        std::shared_ptr<Node>
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
