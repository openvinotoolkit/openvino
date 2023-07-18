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
OPENVINO_SUPPRESS_DEPRECATED_START
class ONNXFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("ONNXFrameworkNode", "util", ov::op::util::FrameworkNode);

    ONNXFrameworkNode(const onnx_import::Node& node) : ONNXFrameworkNode(node, node.get_ng_inputs()) {}

    ONNXFrameworkNode(const onnx_import::Node& node, const OutputVector& inputs)
        : ov::op::util::FrameworkNode(inputs, node.get_outputs_size()),
          m_node(node) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name(node.op_type());
        attrs.set_opset_name(node.domain());
        set_attrs(attrs);
    }

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
    OPENVINO_OP("ONNXSubgraphFrameworkNode", "util", ONNXFrameworkNode);

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
OPENVINO_SUPPRESS_DEPRECATED_END

// Be careful with using protobuf references (also onnx_import::Node) inside NotSupportedONNXNode
// which are inserted into ov::Model due to different lifetime and problematic sharing between dynamic libs.
class NotSupportedONNXNode : public ov::op::util::FrameworkNode {
    static constexpr const char* failed_conversion_key = "onnx::NotSupportedONNXNode::failed_conversion_key";

public:
    OPENVINO_OP("NotSupportedONNXNode", "util", ov::op::util::FrameworkNode);

    NotSupportedONNXNode(const OutputVector& inputs,
                         const size_t output_size,
                         const std::string& domain,
                         const std::string& op_type,
                         const std::string& additional_error_message)
        : ov::op::util::FrameworkNode(inputs, output_size) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_opset_name(domain);
        attrs.set_type_name(op_type);
        attrs[failed_conversion_key] = additional_error_message;
        set_attrs(attrs);
    }

    std::string additional_error_message() const {
        auto attrs = get_attrs();
        return attrs[failed_conversion_key];
    }

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    virtual bool visit_attributes(AttributeVisitor& visitor) override;
};

}  // namespace frontend
}  // namespace ngraph
