//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <onnx_import/onnx_node.hpp>

namespace ngraph
{
namespace frontend
{

NGRAPH_RTTI_DEFINITION(ONNXNode, "__ONNXNode", 1);

std::shared_ptr<Node> ONNXNode::clone_with_new_inputs(const OutputVector& inputs) const
{
    return std::make_shared<ONNXNode>(inputs, node);
}
/*
    OutputVector framework_node_factory (const ngraph::onnx_import::Node& node)
{
    auto ng_node = std::make_shared<ONNXNode>(node.get_ng_inputs(), node.get_outputs_size());
    return ng_node->outputs();
}
 */

} // namespace frontend
} // namespace ngraph
