/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <tensorflow_frontend/frontend.hpp>
#include <tensorflow_frontend/utility.hpp>
#include <tensorflow_frontend/place.hpp>
#include <tensorflow_frontend/model.hpp>

#include "ngraph_conversions.h"
#include "node_context_impl.hpp"
#include "node_context_new.hpp"
#include "decoder_new.hpp"

namespace tensorflow {
class GraphDef;
class NodeDef;
namespace ngraph_bridge {
class GraphIteratorProto;
}
}  // namespace tensorflow

namespace tensorflow {
namespace ngraph_bridge {
class NodeProtoWrapper : public ngraph::frontend::tensorflow::detail::TFNodeDecoder {
    const NodeDef* node_def;
    // const GraphDef* graph_def;
public:
    NodeProtoWrapper(const NodeDef* _node_def) : node_def(_node_def) {
        /*
        // Try to print all existing attributes
        std::cout << " ******** Attributes for " << node_def->name() << " of type " << node_def->op() << "\n";
        for(const auto& a : node_def->attr())
        {
            std::cout << "    " << a.first << ": " << a.second.value_case() << "\n";
        }
        std::cout << " ***** end of attributes ****" << std::endl;
         */
    }

#define GET_ATTR_VALUE(TYPE, FIELD)                                        \
    virtual void getAttrValue2(const char* name, TYPE* x) const override { \
        *x = node_def->attr().at(name).FIELD();                            \
    }
#define GET_ATTR_VALUE_VECTOR(TYPE, FIELD)                                              \
    virtual void getAttrValue2(const char* name, std::vector<TYPE>* x) const override { \
        const auto& list = node_def->attr().at(name).list();                            \
        x->reserve(/*node_def->attr().at(name).FIELD##_size()*/ list.FIELD##_size());   \
        for (size_t i = 0; i < list.FIELD##_size(); ++i) {                              \
            x->push_back(list.FIELD(i));                                                \
        }                                                                               \
    }

    GET_ATTR_VALUE_VECTOR(int32_t, i)
    GET_ATTR_VALUE_VECTOR(int64_t, i)
    GET_ATTR_VALUE_VECTOR(float, f)
    GET_ATTR_VALUE_VECTOR(std::string, s)
    GET_ATTR_VALUE(int32_t, i)
    GET_ATTR_VALUE(int64_t, i)
    GET_ATTR_VALUE(std::string, s)
    GET_ATTR_VALUE(bool, b)
    GET_ATTR_VALUE(float, f)

    virtual void getAttrValue2(const char* name, DataType* x) const override {
        *x = node_def->attr().at(name).type();
    }

    virtual void getAttrValue2(const char* name, ngraph::element::Type* x) const override {
        DataType dt;
        getAttrValue2(name, &dt);
        try {
            ::tensorflow::ngraph_bridge::TFDataTypeToNGraphElementType(dt, x);
        } catch (const std::out_of_range&) {
            throw errors::Unimplemented("Failed to convert TF data type: " + DataType_Name(dt));
        }
    }

    virtual void getAttrValue2(const char* name, ngraph::PartialShape* x) const override {
        TFTensorShapeToNGraphShape(node_def->attr().at(name).shape(), x);
    }

    // a way to read Const value as a tensor
    virtual void getAttrValue2(const char* name,
                               ngraph::frontend::tensorflow::detail::TensorWrapper** x) const override {
        // TODO: use std::shared_ptr! memory is lost!
        *x = new ngraph::frontend::tensorflow::detail::TensorWrapper(&node_def->attr().at(name).tensor());
    }

    virtual std::string op() const override {
        return node_def->op();
    }

    virtual unsigned int num_inputs() const override {
        return node_def->input_size();
    }

    virtual std::string name() const override {
        return node_def->name();
    }

    virtual std::string type_string() const override {
        return node_def->op();
    }

    Status input_node(size_t index, std::string* name) const override {
        NGRAPH_TF_FE_NOT_IMPLEMENTED;
    }

    Status input_node(size_t index, std::string* name, size_t* outputPortIndex) const override {
        std::string input_name = node_def->input(index);
        // TODO: Implement full logic to detect only the last : as a separator, consult with TF
        auto portDelimPos = input_name.find(':');
        if (portDelimPos != std::string::npos) {
            *name = input_name.substr(0, portDelimPos);
            *outputPortIndex = std::stoi(input_name.substr(portDelimPos));
        }
        *name = input_name;
        *outputPortIndex = 0;
        return Status::OK();
    }

    virtual DataType input_type(size_t index) const override {
        NGRAPH_TF_FE_NOT_IMPLEMENTED;
    }
    virtual DataType output_type(size_t index) const override {
        NGRAPH_TF_FE_NOT_IMPLEMENTED;
    }

    virtual bool IsSink() const override {
        // TODO: recognize special op in TF runtime; don't know similar node for proto graph representation
        return false;
    }

    virtual bool IsSource() const override {
        // TODO: populate with other source operation types
        return node_def->op() == "Placeholder";
    }

    virtual bool IsControlFlow() const override {
        // TODO
        return false;
    }

    virtual std::string DebugString() const override {
        return node_def->op() + "(with name " + node_def->name() + ")";
        // return node_def->DebugString();
    }

    virtual bool IsArg() const override {
        // TODO
        return IsSource();
    }

    virtual bool IsRetval() const override {
        // TODO
        return IsSink();
    }
};

class GraphIteratorProto : public ::ngraph::frontend::GraphIterator {
    std::vector<const ::tensorflow::NodeDef*> nodes;
    size_t node_index = 0;

public:
    GraphIteratorProto(const GraphDef* _graph) {
        // TODO: Sort topologicaly nodes from the graph
        nodes.resize(_graph->node_size());
        for (size_t i = 0; i < nodes.size(); ++i)
            nodes[i] = &_graph->node(i);
    }

    GraphIteratorProto(const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _sorted_nodes) {
        nodes.resize(_sorted_nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
            nodes[i] = _sorted_nodes[i].get();
    }

    /// Set iterator to the start position
    virtual void reset() override {
        node_index = 0;
    }

    virtual size_t size() const override {
        return nodes.size();
    }

    /// Moves to the next node in the graph
    virtual void next() override {
        node_index++;
    }

    virtual bool is_end() const override {
        return node_index >= nodes.size();
    }

    /// Return NodeContext for the current node that iterator points to
    virtual std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder> get() const override {
        return std::make_shared<NodeProtoWrapper>(nodes[node_index]);
    }

    virtual std::shared_ptr<ngraph::frontend::DecoderBase> get_new() const override {
        return std::make_shared<::ngraph::frontend::DecoderTFProto>(nodes[node_index]);
    }
};

}  // namespace ngraph_bridge
}  // namespace tensorflow
