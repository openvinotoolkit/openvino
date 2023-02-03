//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
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

#pragma once

#include <core/graph.hpp>
#include <ngraph/function.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/visibility.hpp>
#include <onnx_import/core/node.hpp>
#include <openvino/op/util/framework_node.hpp>

namespace ONNX_NAMESPACE {
// forward declaration
class ModelProto;
}  // namespace ONNX_NAMESPACE

namespace ngraph {
namespace onnx_import {
class Model;
}

namespace frontend {
class ONNXFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_RTTI("ONNXFrameworkNode", "0");

    ONNXFrameworkNode(const onnx_import::Node& node)
        : ov::op::util::FrameworkNode(node.get_ng_inputs(), node.get_outputs_size()),
          m_node(node) {}

    ONNXFrameworkNode(const onnx_import::Node& node, const OutputVector& inputs)
        : ov::op::util::FrameworkNode(inputs, node.get_outputs_size()),
          m_node(node) {}

    OutputVector get_ng_nodes(const std::shared_ptr<onnx_import::Graph>& graph) const {
        OutputVector ng_nodes{graph->make_ng_nodes(m_node)};
        if (ng_nodes.size() > get_output_size()) {
            ng_nodes.resize(get_output_size());
        }
        return ng_nodes;
    }

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    virtual bool visit_attributes(AttributeVisitor& visitor) override {
        // TODO: implement reading as well, now it work for serialization only
        std::string domain = m_node.domain();
        std::string op_type = m_node.op_type();
        visitor.on_attribute("ONNX_META_domain", domain);
        visitor.on_attribute("ONNX_META_type", op_type);
        return true;
    }

protected:
    onnx_import::Node m_node;
};

class ONNXSubgraphFrameworkNode : public ONNXFrameworkNode {
public:
    OPENVINO_RTTI("ONNXSubgraphFrameworkNode", "0");

    ONNXSubgraphFrameworkNode(const onnx_import::Node& node,
                              const std::vector<std::shared_ptr<Function>>& functions,
                              const OutputVector& inputs)
        : ONNXFrameworkNode(node, inputs),
          m_functions(functions) {}

    void infer_inputs_from_parent() {
        for (auto& subgraph : m_node.get_subgraphs())
            subgraph.second->infer_inputs_from_parent();
    }

    const std::vector<std::shared_ptr<Function>>& get_subgraph_functions() const {
        return m_functions;
    }

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

private:
    std::vector<std::shared_ptr<Function>> m_functions;
};

}  // namespace frontend
}  // namespace ngraph
