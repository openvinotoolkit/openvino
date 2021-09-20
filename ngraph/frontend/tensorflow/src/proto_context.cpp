// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/place.hpp>

#include "default_opset.h"
#include "graph.hpp"
#include "graph.pb.h"
#include "ngraph_builder.h"
#include "ngraph_conversions.h"
#include "node_context_impl.hpp"
#include "tensor.pb.h"

namespace ngraph {
namespace frontend {
namespace tensorflow {
namespace detail {
#if 0
    using ::tensorflow::NodeDef;
    using ::tensorflow::GraphDef;
    using ::tensorflow::DataType;
    using ::tensorflow::Status;
    using ::tensorflow::ngraph_bridge::TFDataTypeToNGraphElementType;
    using ::tensorflow::ngraph_bridge::TFTensorShapeToNGraphShape;
    using ::tensorflow::errors;
    namespace ng = ngraph;

class NodeProtoWrapper : public ::tensorflow::TFNodeDecoder {
    const NodeDef *node_def;
    const GraphDef *graph_def;
public:

    NodeProtoWrapper(const NodeDef *_node_def, const GraphDef *_graph_def) :
            node_def(_node_def), graph_def(_graph_def)
            {

            }

#    define TEMPLATE_EXPLICIT_SPECIALIZATION(TYPE, FIELD)                      \
        virtual void getAttrValue2(const char* name, TYPE* x) const override { \
            *x = node_def->attr().at(name).FIELD();                            \
        }
#    define TEMPLATE_EXPLICIT_SPECIALIZATION_V(TYPE, FIELD)                                 \
        virtual void getAttrValue2(const char* name, std::vector<TYPE>* x) const override { \
            const auto& list = node_def->attr().at(name).list();                            \
            x->reserve(/*node_def->attr().at(name).FIELD##_size()*/ list.FIELD##_size());   \
            for (size_t i = 0; i < list.FIELD##_size(); ++i) {                              \
                x->push_back(list.FIELD(i));                                                \
            }                                                                               \
        }

    TEMPLATE_EXPLICIT_SPECIALIZATION_V(int32_t, i)

    TEMPLATE_EXPLICIT_SPECIALIZATION_V(float, f)
    //virtual void getAttrValue2 (const char* name, std::vector<int32_t>* x) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }
    //virtual void getAttrValue2 (const char* name, std::vector<float>* x) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }
    TEMPLATE_EXPLICIT_SPECIALIZATION(int32_t, i)

    virtual void getAttrValue2(const char *name, DataType *x) const override {
        *x = node_def->attr().at(name).type();
    }

    virtual void getAttrValue2(const char *name, ngraph::element::Type *x) const override {
        DataType dt;
        getAttrValue2(name, &dt);
        try {
            TFDataTypeToNGraphElementType(dt, x);
        }
        catch (const std::out_of_range &) {
            throw errors::Unimplemented("Failed to convert TF data type: " +
                                        DataType_Name(dt));
        }
    }

    virtual void getAttrValue2(const char *name, ngraph::PartialShape *x) const override {
        TFTensorShapeToNGraphShape(node_def->attr().at(name).shape(), x);
    }

    TEMPLATE_EXPLICIT_SPECIALIZATION(std::string, s)

    TEMPLATE_EXPLICIT_SPECIALIZATION(bool, b)

    TEMPLATE_EXPLICIT_SPECIALIZATION(long int, i)

    TEMPLATE_EXPLICIT_SPECIALIZATION(float, f)

    virtual void
    getAttrValue2(const char *name,
                 std::vector<std::string> *x) const override {NGRAPH_TF_FE_NOT_IMPLEMENTED; }

    // a way to read Const value as a tensor
    virtual void getAttrValue2(const char *name, ngraph::frontend::tensorflow::detail::TensorWrapper **x) const override {
        // TODO: use std::shared_ptr! memory is lost!
        *x = new ngraph::frontend::tensorflow::detail::TensorWrapper(&node_def->attr().at(name).tensor());
    }

    virtual std::string op() const override {
        return node_def->op();
    }

    virtual unsigned int num_inputs() const override { return node_def->input_size(); }

    virtual std::string name() const override {
        return node_def->name();
    }

    virtual std::string type_string() const override {
        return node_def->op();
    }

    Status input_node(size_t index, std::string *name) const {NGRAPH_TF_FE_NOT_IMPLEMENTED; }

    Status input_node(size_t index, std::string *name, size_t *outputPortIndex) const {
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

    virtual DataType input_type(size_t index) const override {NGRAPH_TF_FE_NOT_IMPLEMENTED; }

    virtual DataType output_type(size_t index) const override {NGRAPH_TF_FE_NOT_IMPLEMENTED; }

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


class GraphIteratorProto : public ng::frontend::tensorflow::GraphIterator {
    const GraphDef *graph;
    size_t node_index = 0;

public:

    GraphIteratorProto(const GraphDef *_graph) :
            graph(_graph) {
    }

    /// Set iterator to the start position
    virtual void reset() {
        node_index = 0;
    }

    virtual size_t size() const {
        return graph->node_size();
    }

    /// Moves to the next node in the graph
    virtual void next() {
        node_index++;
    }

    virtual bool is_end() const {
        return node_index >= graph->node_size();
    }

    /// Return NodeContext for the current node that iterator points to
    virtual std::shared_ptr<ng::frontend::tensorflow::detail::TFNodeDecoder> get() const {
        return std::make_shared<NodeProtoWrapper>(&graph->node(node_index), graph);

    }
};

#endif
NodeContext::NodeContext(const OutputVector& _ng_inputs,
                         std::shared_ptr<detail::TFNodeDecoder> _decoder,
                         const std::map<std::string, ngraph::PartialShape>& overridden_shapes,
                         const std::vector<ngraph::PartialShape>& indexed_shapes)
    : m_ng_inputs(_ng_inputs),
      m_decoder(_decoder),
      m_overridden_shapes(overridden_shapes),
      m_indexed_shapes(indexed_shapes) {}

NodeContext::NodeContext(const OutputVector& _ng_inputs,
                         std::shared_ptr<detail::TFNodeDecoder> _decoder,
                         const std::vector<Place::Ptr>& _inputs)
    : m_indexed_shapes{} {
    m_ng_inputs = _ng_inputs;
    m_decoder = _decoder;

    for (const auto& inp : _inputs) {
        const auto& input_tensor_place = std::dynamic_pointer_cast<TensorPlaceTF>(inp);
        m_overridden_shapes[input_tensor_place->get_names()[0]] = input_tensor_place->get_partial_shape();
    }
}

size_t NodeContext::get_ng_input_size() const {
    return m_ng_inputs.size();
}

/// Returns a vector of already converted inputs for this node
const OutputVector& NodeContext::get_ng_inputs() const {
    return m_ng_inputs;
}

Output<Node> NodeContext::get_ng_input(size_t input_port) const {
    return m_ng_inputs[input_port];
}

std::string NodeContext::get_op_type() const {
    return m_decoder->op();
}

std::vector<std::string> NodeContext::get_output_names() const {
    // TODO
    throw "Not implemented";
}

std::vector<std::string> NodeContext::get_names() const {
    std::vector<std::string> names;
    names.push_back(m_decoder->name());
    return names;
}

std::string NodeContext::get_name() const {
    return get_names()[0];
}

/// Temporary method for the transition period during migration to NodeContext
// TODO: Remove this method and port all dependent code to the remaining methods
const detail::TFNodeDecoder* NodeContext::_get_decoder() const {
    return m_decoder.get();
}

template <typename T>
T NodeContext::get_attribute(const std::string& name) const {
    try {
        T result;
        m_decoder->getAttrValue2(name.c_str(), &result);
        // TODO: no real processing of case when there is no default: getAttrValue will provide default even you don't
        // need it
        return result;
    } catch (...) {
        std::cerr << "[ ERROR ] When accecing attribute '" << name << "' value.\n";
        throw;
    }
}

template <typename T>
T NodeContext::get_attribute(const std::string& name, const T& default_value) const {
    T result;
    try {
        m_decoder->getAttrValue2(name.c_str(), &result);
    } catch (...)  // TODO: replace by more narrow filter
    {
        result = default_value;
    }
    return result;
}

// Meta-attributes like op type, domain, version -- some FW specific but common for all operations properties

const std::map<std::string, ngraph::PartialShape>& NodeContext::get_overridden_shapes() const {
    return m_overridden_shapes;
}

const std::vector<ngraph::PartialShape>& NodeContext::get_indexed_shapes() const {
    return m_indexed_shapes;
}

#define TEMPLATE_EXPLICIT_SPECIALIZATION(T)                             \
    template T NodeContext::get_attribute<T>(const std::string&) const; \
    template T NodeContext::get_attribute<T>(const std::string&, const T&) const;

#define TEMPLATE_EXPLICIT_SPECIALIZATION_V(T)                                                     \
    template std::vector<T> NodeContext::get_attribute<std::vector<T>>(const std::string&) const; \
    template std::vector<T> NodeContext::get_attribute<std::vector<T>>(const std::string&, const std::vector<T>&) const;

TEMPLATE_EXPLICIT_SPECIALIZATION_V(int32_t)
TEMPLATE_EXPLICIT_SPECIALIZATION_V(int64_t)
TEMPLATE_EXPLICIT_SPECIALIZATION_V(float)
TEMPLATE_EXPLICIT_SPECIALIZATION(int32_t)
TEMPLATE_EXPLICIT_SPECIALIZATION(int64_t)

TEMPLATE_EXPLICIT_SPECIALIZATION(ngraph::element::Type)

TEMPLATE_EXPLICIT_SPECIALIZATION(ngraph::PartialShape)
TEMPLATE_EXPLICIT_SPECIALIZATION(std::string)
TEMPLATE_EXPLICIT_SPECIALIZATION(bool)
TEMPLATE_EXPLICIT_SPECIALIZATION(float)

TEMPLATE_EXPLICIT_SPECIALIZATION_V(std::string)

}  // namespace detail
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ngraph
