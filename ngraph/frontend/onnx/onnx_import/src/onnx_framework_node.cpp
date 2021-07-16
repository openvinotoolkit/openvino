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

#include <onnx_import/onnx_framework_node.hpp>

namespace ngraph
{
    namespace frontend
    {
        NGRAPH_RTTI_DEFINITION(ONNXFrameworkNode, "ONNXFrameworkNode", 1);

        std::shared_ptr<Node>
            ONNXFrameworkNode::clone_with_new_inputs(const OutputVector& inputs) const
        {
            return std::make_shared<ONNXFrameworkNode>(m_node, inputs);
        }

        NGRAPH_RTTI_DEFINITION(ONNXSubgraphFrameworkNode, "ONNXSubgraphFrameworkNode", 1);

    } // namespace frontend
} // namespace ngraph
