//*****************************************************************************
// Copyright (C) 2017-2024 Intel Corporation
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

#include "core/graph.hpp"
#include "core/node.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ONNX_NAMESPACE {
// forward declaration
class ModelProto;
}  // namespace ONNX_NAMESPACE

namespace ov {
namespace frontend {
namespace onnx {
class Model;

class ONNXFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("ONNXFrameworkNode", "util", ov::op::util::FrameworkNode);

    ONNXFrameworkNode(const ov::frontend::onnx::Node& node) : ONNXFrameworkNode(node, node.get_ov_inputs()) {}

    ONNXFrameworkNode(const ov::frontend::onnx::Node& node, const ov::OutputVector& inputs)
        : ov::op::util::FrameworkNode(inputs, node.get_outputs_size()),
          m_node(node) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name(node.op_type());
        attrs.set_opset_name(node.domain());
        set_attrs(attrs);
    }

    ov::OutputVector get_ov_nodes(const std::shared_ptr<ov::frontend::onnx::Graph>& graph) const {
        ov::OutputVector ov_nodes{graph->make_ov_nodes(m_node)};
        if (ov_nodes.size() > get_output_size()) {
            ov_nodes.resize(get_output_size());
        }
        return ov_nodes;
    }

    virtual std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override;

    virtual bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // TODO: implement reading as well, now it work for serialization only
        std::string domain = m_node.domain();
        std::string op_type = m_node.op_type();
        visitor.on_attribute("ONNX_META_domain", domain);
        visitor.on_attribute("ONNX_META_type", op_type);
        return true;
    }

protected:
    ov::frontend::onnx::Node m_node;
};

class ONNXSubgraphFrameworkNode : public ONNXFrameworkNode {
public:
    OPENVINO_OP("ONNXSubgraphFrameworkNode", "util", ONNXFrameworkNode);

    ONNXSubgraphFrameworkNode(const ov::frontend::onnx::Node& node,
                              const std::vector<std::shared_ptr<ov::Model>>& models,
                              const ov::OutputVector& inputs)
        : ONNXFrameworkNode(node, inputs),
          m_models(models) {}

    void infer_inputs_from_parent() {
        for (auto& subgraph : m_node.get_subgraphs())
            subgraph.second->infer_inputs_from_parent();
    }

    const std::vector<std::shared_ptr<ov::Model>>& get_subgraph_models() const {
        return m_models;
    }

    virtual std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override;

private:
    std::vector<std::shared_ptr<ov::Model>> m_models;
};

// Be careful with using protobuf references (also ov::frontend::onnx::Node) inside NotSupportedONNXNode
// which are inserted into ov::Model due to different lifetime and problematic sharing between dynamic libs.
class NotSupportedONNXNode : public ov::op::util::FrameworkNode {
    static constexpr const char* failed_conversion_key = "onnx::NotSupportedONNXNode::failed_conversion_key";

public:
    OPENVINO_OP("NotSupportedONNXNode", "util", ov::op::util::FrameworkNode);

    NotSupportedONNXNode(const ov::OutputVector& inputs,
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

    virtual std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override;
    virtual bool visit_attributes(ov::AttributeVisitor& visitor) override;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
