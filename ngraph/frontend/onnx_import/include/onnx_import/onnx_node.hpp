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

#include <ngraph/visibility.hpp>
#include <onnx_import/core/node.hpp>
#include "framework_node.hpp"

namespace ONNX_NAMESPACE
{
    // forward declaration
    class ModelProto;
}

namespace ngraph
{

namespace onnx_import
{
    class Model;
}

namespace frontend
{

class NGRAPH_API ONNXNode : public FrameworkNode
{
    NGRAPH_RTTI_DECLARATION;

    onnx_import::Node node;

    std::shared_ptr<onnx_import::Graph> graph;
    std::shared_ptr<onnx_import::Model> model;
    std::shared_ptr<const ONNX_NAMESPACE::ModelProto> model_proto;

public:

    ONNXNode (const onnx_import::Node& _node) :
        FrameworkNode(_node.get_ng_inputs(), _node.get_outputs_size()),
        node(_node)
    {
    }

    ONNXNode (const OutputVector& _inputs, const onnx_import::Node& _node) :
        FrameworkNode(_inputs, _node.get_outputs_size()),
        node(_node)
    {
    }

    void set_onnx_graph (std::shared_ptr<onnx_import::Graph> _graph) { graph = _graph; }
    void set_onnx_model (std::shared_ptr<onnx_import::Model> _model) { model = _model; }
    void set_onnx_model_proto (std::shared_ptr<const ONNX_NAMESPACE::ModelProto> _model_proto) { model_proto = _model_proto; }

    std::shared_ptr<onnx_import::Graph> get_onnx_graph () const { return graph; }
    //void get_onnx_model (std::shared_ptr<onnx_import::Model> _model) { model = _model; }
    //void get_onnx_model_proto (std::shared_ptr<const ONNX_NAMESPACE::ModelProto> _model_proto) { model_proto = _model_proto; }

    const onnx_import::Node& get_onnx_node () const { return node; }

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    virtual bool visit_attributes(AttributeVisitor& visitor) override
    {
        // TODO: implement reading as well, now it work for serialization only
        std::string domain = node.domain();
        std::string op_type = node.op_type();
        visitor.on_attribute("ONNX_META_domain", domain);
        visitor.on_attribute("ONNX_META_type", op_type);
        return true;
    }
};

inline OutputVector framework_node_factory (const ngraph::onnx_import::Node& node)
{
    auto ng_node = std::make_shared<ONNXNode>(node);
    return ng_node->outputs();
}

} // namespace frontend
} // namespace ngraph
