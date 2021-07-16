// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include "detail/subgraph_extraction.hpp"
#include "ngraph/log.hpp"
#include "onnx_common/parser.hpp"
#include "onnx_common/utils.hpp"
#include "onnx_editor/edge_mapper.hpp"
#include "onnx_editor/editor.hpp"
#include "onnx_import/utils/onnx_internal.hpp"

using namespace ngraph;
using namespace ngraph::onnx_editor;

namespace
{
    using namespace ONNX_NAMESPACE;

    ValueInfoProto* find_graph_input(GraphProto& graph, const std::string& name)
    {
        for (int i = 0; i < graph.input_size(); ++i)
        {
            auto* input_desc = graph.mutable_input(i);
            if (input_desc->has_name() && input_desc->name() == name)
            {
                return input_desc;
            }
        }

        return nullptr;
    }

    TensorProto* find_graph_initializer(GraphProto& graph, const std::string& name)
    {
        for (int i = 0; i < graph.initializer_size(); ++i)
        {
            auto* initializer_desc = graph.mutable_initializer(i);
            if (initializer_desc->has_name() && initializer_desc->name() == name)
                return initializer_desc;
        }

        return nullptr;
    }

    void modify_input_type(ValueInfoProto& onnx_input, const element::Type_t elem_type)
    {
        if (!onnx_input.has_type())
        {
            throw ngraph_error(
                "The input is malformed - it doesn't contain the 'type' field. Cannot change the "
                "data type. Input name: " +
                onnx_input.name());
        }

        auto* type_proto = onnx_input.mutable_type();
        if (!type_proto->has_tensor_type())
        {
            throw ngraph_error(
                "The input is malformed - it doesn't contain the 'tensor_type' field. Cannot "
                "change the data type. Input name: " +
                onnx_input.name());
        }

        auto* tensor_type = type_proto->mutable_tensor_type();

        if (onnx_common::is_supported_ng_type(elem_type))
        {
            tensor_type->set_elem_type(onnx_common::ng_to_onnx_data_type(elem_type));
        }
        else
        {
            throw ngraph_error("The input type for input '" + onnx_input.name() +
                               "' cannot be set to: " + element::Type(elem_type).get_type_name() +
                               ". This type is not allowed in ONNX.");
        }
    }

    void add_dim_to_onnx_shape(const Dimension& dim, ONNX_NAMESPACE::TensorShapeProto& onnx_shape)
    {
        auto* new_dim = onnx_shape.add_dim();
        if (dim.is_static())
        {
            new_dim->set_dim_value(dim.get_length());
        }
        else
        {
            // nGraph Dimension is also considered dynamic if it represents a constrained range
            // of allowed values as well as if it's unconstrained at all. ONNX cannot represent
            // ranged dimensions so this might not be 100% accurate. The modified ONNX model will
            // always have a fully dynamic dimension in this case.
            new_dim->set_dim_param("__dynamic_dimension__");
        }
    }

    void modify_input_shape(ValueInfoProto& onnx_input, const PartialShape& new_shape)
    {
        if (!onnx_input.has_type())
        {
            throw ngraph_error(
                "The input is malformed - it doesn't contain the 'type' field. Cannot change the "
                "input shape. Input name: " +
                onnx_input.name());
        }

        auto* type_proto = onnx_input.mutable_type();
        if (!type_proto->has_tensor_type())
        {
            throw ngraph_error(
                "The input is malformed - it doesn't contain the 'tensor_type' field. Cannot "
                "change the input shape. Input name: " +
                onnx_input.name());
        }

        auto* tensor_type = type_proto->mutable_tensor_type();
        if (new_shape.rank().is_dynamic())
        {
            tensor_type->clear_shape();
        }
        else
        {
            // make a copy intentionally, in case of an exception the original model is not modified
            auto new_onnx_shape = tensor_type->shape();
            new_onnx_shape.clear_dim();

            for (const auto& dim : static_cast<std::vector<Dimension>>(new_shape))
            {
                add_dim_to_onnx_shape(dim, new_onnx_shape);
            }

            *(tensor_type->mutable_shape()) = std::move(new_onnx_shape);
        }
    }

    template <typename T>
    std::string extract_name(const T& input_or_initializer)
    {
        return input_or_initializer.name();
    };

    void modify_initializer(TensorProto& initializer,
                            const std::string& name,
                            const std::shared_ptr<ngraph::op::Constant> values,
                            ValueInfoProto* input)
    {
        const auto elem_type = values->get_element_type();
        if (!onnx_common::is_supported_ng_type(elem_type))
        {
            throw ngraph_error("Initializer '" + name + "' type cannot be set to: " +
                               element::Type(elem_type).get_type_name() +
                               ". This type is not allowed in ONNX.");
        }

        initializer.Clear();

        initializer.set_name(name);
        initializer.set_data_type(onnx_common::ng_to_onnx_data_type(values->get_element_type()));

        for (const auto& dim : values->get_shape())
        {
            initializer.add_dims(dim);
        }

        const auto data_size_in_bytes = shape_size(values->get_shape()) *
                                        onnx_common::get_onnx_data_size(initializer.data_type());
        initializer.set_raw_data(values->get_data_ptr(), data_size_in_bytes);

        // update input with type and shape of initializer
        if (input)
        {
            auto tensor_type = input->mutable_type()->mutable_tensor_type();
            TensorShapeProto shape;
            for (int i = 0; i < initializer.dims_size(); ++i)
            {
                shape.add_dim()->set_dim_value(initializer.dims(i));
            }
            *tensor_type->mutable_shape() = std::move(shape);
            tensor_type->set_elem_type(initializer.data_type());
        }
    }
} // namespace

/// \brief A helper class used to hold the ModelProto object as its field
struct onnx_editor::ONNXModelEditor::Impl
{
    ONNX_NAMESPACE::ModelProto m_model_proto;
    EdgeMapper m_edge_mapper;
    bool m_is_mapper_updated = false;

    Impl() = delete;

    Impl(const std::string& model_path)
        : m_model_proto{onnx_common::parse_from_file(model_path)}
    {
    }

    void infer_shapes() { ONNX_NAMESPACE::shape_inference::InferShapes(m_model_proto); }
    void remove_shape_inference_info() { m_model_proto.mutable_graph()->clear_value_info(); }
};

onnx_editor::ONNXModelEditor::ONNXModelEditor(const std::string& model_path)
    : m_model_path{model_path}
    , m_pimpl{new ONNXModelEditor::Impl{model_path}, [](Impl* impl) { delete impl; }}
{
}

const std::string& onnx_editor::ONNXModelEditor::model_path() const
{
    return m_model_path;
}

void onnx_editor::ONNXModelEditor::serialize(const std::string& out_file_path) const
{
    std::ofstream out_file{out_file_path, std::ios::out | std::ios::binary};

    if (!out_file.is_open())
    {
        throw ngraph_error("Could not open the file: " + out_file_path);
    };

    if (!m_pimpl->m_model_proto.SerializeToOstream(&out_file))
    {
        throw ngraph_error("Could not serialize the model to: " + out_file_path);
    }
    else
    {
        out_file.close();
    }
}

void onnx_editor::ONNXModelEditor::set_input_types(
    const std::map<std::string, element::Type_t>& input_types)
{
    auto* onnx_graph = m_pimpl->m_model_proto.mutable_graph();

    for (const auto& input_desc : input_types)
    {
        auto* onnx_input = find_graph_input(*onnx_graph, input_desc.first);
        if (onnx_input != nullptr)
        {
            modify_input_type(*onnx_input, input_desc.second);
        }
        else
        {
            throw ngraph_error(
                "Could not set a custom element type for input: " + input_desc.first +
                ". Such input was not found in the original ONNX model.");
        }
    }
}

void onnx_editor::ONNXModelEditor::set_input_shapes(
    const std::map<std::string, ngraph::PartialShape>& input_shapes)
{
    auto* onnx_graph = m_pimpl->m_model_proto.mutable_graph();

    for (const auto& input_desc : input_shapes)
    {
        auto* onnx_input = find_graph_input(*onnx_graph, input_desc.first);
        if (onnx_input != nullptr)
        {
            modify_input_shape(*onnx_input, input_desc.second);
        }
        else
        {
            throw ngraph_error("Could not set custom shape for input: " + input_desc.first +
                               ". Such input was not found in the original ONNX model.");
        }
    }
}

void onnx_editor::ONNXModelEditor::cut_graph_fragment(const std::vector<InputEdge>& inputs,
                                                      const std::vector<OutputEdge>& outputs)
{
    if (inputs.empty() && outputs.empty())
    {
        return;
    }

    m_pimpl->infer_shapes();

    SubgraphExtractor editor{*(m_pimpl->m_model_proto.mutable_graph())};
    editor.add_new_inputs(inputs);
    editor.add_new_outputs(outputs);
    editor.extract_subgraph(outputs);

    m_pimpl->remove_shape_inference_info();
    m_pimpl->m_is_mapper_updated = false;
}

std::vector<std::string> onnx_editor::ONNXModelEditor::model_inputs() const
{
    const auto& graph = m_pimpl->m_model_proto.graph();

    std::vector<std::string> inputs_and_initializers;
    inputs_and_initializers.reserve(graph.input_size() + graph.initializer_size());

    std::transform(graph.input().begin(),
                   graph.input().end(),
                   std::back_inserter(inputs_and_initializers),
                   extract_name<ONNX_NAMESPACE::ValueInfoProto>);

    std::transform(graph.initializer().begin(),
                   graph.initializer().end(),
                   std::back_inserter(inputs_and_initializers),
                   extract_name<ONNX_NAMESPACE::TensorProto>);

    return inputs_and_initializers;
}

std::string onnx_editor::ONNXModelEditor::model_string() const
{
    return m_pimpl->m_model_proto.SerializeAsString();
}

std::shared_ptr<Function> onnx_editor::ONNXModelEditor::get_function() const
{
    return onnx_import::detail::import_onnx_model(m_pimpl->m_model_proto, m_model_path);
}

void onnx_editor::ONNXModelEditor::set_input_values(
    const std::map<std::string, std::shared_ptr<ngraph::op::Constant>>& input_values)
{
    auto onnx_graph = m_pimpl->m_model_proto.mutable_graph();

    for (const auto& input : input_values)
    {
        auto& name = input.first;
        auto& values = input.second;

        auto onnx_input = find_graph_input(*onnx_graph, name);
        auto onnx_initializer = find_graph_initializer(*onnx_graph, name);

        if (!onnx_initializer && !onnx_input)
        {
            NGRAPH_INFO << "There is no input nor initializer named '" << name
                        << "' in original model '" << m_model_path << "'.";
        }

        if (!onnx_initializer)
        {
            onnx_initializer = onnx_graph->add_initializer();
        }

        modify_initializer(*onnx_initializer, name, values, onnx_input);
    }
}

void onnx_editor::ONNXModelEditor::update_mapper_if_needed() const
{
    if (!m_pimpl->m_is_mapper_updated)
    {
        m_pimpl->m_edge_mapper = EdgeMapper(m_pimpl->m_model_proto.graph());
    }
    m_pimpl->m_is_mapper_updated = true;
}

InputEdge onnx_editor::ONNXModelEditor::find_input_edge(const EditorNode& node,
                                                        const EditorInput& input) const
{
    update_mapper_if_needed();
    return m_pimpl->m_edge_mapper.find_input_edge(node, input);
}

OutputEdge onnx_editor::ONNXModelEditor::find_output_edge(const EditorNode& node,
                                                          const EditorOutput& input) const
{
    update_mapper_if_needed();
    return m_pimpl->m_edge_mapper.find_output_edge(node, input);
}

OutputEdge onnx_editor::ONNXModelEditor::find_output_edge(const std::string& output_name) const
{
    update_mapper_if_needed();
    return m_pimpl->m_edge_mapper.find_output_edge(output_name);
}

std::vector<InputEdge>
    onnx_editor::ONNXModelEditor::find_output_consumers(const std::string& output_name) const
{
    update_mapper_if_needed();
    return m_pimpl->m_edge_mapper.find_output_consumers(output_name);
}

bool onnx_editor::ONNXModelEditor::is_correct_and_unambiguous_node(const EditorNode& node) const
{
    update_mapper_if_needed();
    return m_pimpl->m_edge_mapper.is_correct_and_unambiguous_node(node);
}
