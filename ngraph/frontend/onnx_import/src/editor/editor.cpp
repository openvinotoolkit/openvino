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

#include <fstream>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include "ngraph/log.hpp"
#include "onnx_import/editor/editor.hpp"
#include "utils/common.hpp"
#include "utils/parser.hpp"

using namespace ngraph;

namespace
{
    using namespace ONNX_NAMESPACE;

    const std::map<element::Type_t, TensorProto_DataType> NG_2_ONNX_TYPES = {
        {element::Type_t::bf16, TensorProto_DataType::TensorProto_DataType_BFLOAT16},
        {element::Type_t::f16, TensorProto_DataType::TensorProto_DataType_FLOAT16},
        {element::Type_t::f32, TensorProto_DataType::TensorProto_DataType_FLOAT},
        {element::Type_t::f64, TensorProto_DataType::TensorProto_DataType_DOUBLE},
        {element::Type_t::i8, TensorProto_DataType::TensorProto_DataType_INT8},
        {element::Type_t::i16, TensorProto_DataType::TensorProto_DataType_INT16},
        {element::Type_t::i32, TensorProto_DataType::TensorProto_DataType_INT32},
        {element::Type_t::i64, TensorProto_DataType::TensorProto_DataType_INT64},
        {element::Type_t::u8, TensorProto_DataType::TensorProto_DataType_UINT8},
        {element::Type_t::u16, TensorProto_DataType::TensorProto_DataType_UINT16},
        {element::Type_t::u32, TensorProto_DataType::TensorProto_DataType_UINT32},
        {element::Type_t::u64, TensorProto_DataType::TensorProto_DataType_UINT64},
    };

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
        if (NG_2_ONNX_TYPES.count(elem_type) == 0)
        {
            throw ngraph_error("The input type for input '" + onnx_input.name() +
                               "' cannot be set to: " + element::Type(elem_type).get_type_name() +
                               ". This type is not allowed in ONNX.");
        }
        else
        {
            tensor_type->set_elem_type(NG_2_ONNX_TYPES.at(elem_type));
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
        if (NG_2_ONNX_TYPES.count(elem_type) == 0)
        {
            throw ngraph_error("Initializer '" + name + "' type cannot be set to: " +
                               element::Type(elem_type).get_type_name() +
                               ". This type is not allowed in ONNX.");
        }

        initializer.Clear();

        initializer.set_name(name);
        initializer.set_data_type(NG_2_ONNX_TYPES.at(values->get_element_type()));

        for (const auto& dim : values->get_shape())
        {
            initializer.add_dims(dim);
        }

        const auto data_size_in_bytes =
            shape_size(values->get_shape()) *
            onnx_import::common::get_onnx_data_size(initializer.data_type());
        initializer.set_raw_data(values->get_data_ptr(), data_size_in_bytes);

        // update input with type and shape of initializer
        if (input)
        {
            auto tensor_type = input->mutable_type()->mutable_tensor_type();
            TensorShapeProto shape;
            for (size_t i = 0; i < initializer.dims_size(); ++i)
            {
                shape.add_dim()->set_dim_value(initializer.dims(i));
            }
            *tensor_type->mutable_shape() = std::move(shape);
            tensor_type->set_elem_type(initializer.data_type());
        }
    }
} // namespace

/// \brief A helper class used to hold the ModelProto object as its field
struct onnx_import::ONNXModelEditor::Impl
{
    ONNX_NAMESPACE::ModelProto m_model_proto;

    Impl() = delete;

    Impl(const std::string& model_path)
        : m_model_proto{std::move(parse_from_file(model_path))}
    {
    }

    void infer_shapes() { ONNX_NAMESPACE::shape_inference::InferShapes(m_model_proto); }
    void remove_shape_inference_info() { m_model_proto.mutable_graph()->clear_value_info(); }
};

onnx_import::ONNXModelEditor::ONNXModelEditor(const std::string& model_path)
    : m_pimpl{new ONNXModelEditor::Impl{model_path}, [](Impl* impl) { delete impl; }}
    , m_model_path{model_path}
{
}

ONNX_NAMESPACE::ModelProto& onnx_import::ONNXModelEditor::model() const
{
    return m_pimpl->m_model_proto;
}

const std::string& onnx_import::ONNXModelEditor::model_path() const
{
    return m_model_path;
}

void onnx_import::ONNXModelEditor::serialize(const std::string& out_file_path) const
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

void onnx_import::ONNXModelEditor::set_input_types(
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

void onnx_import::ONNXModelEditor::set_input_shapes(
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

void onnx_import::ONNXModelEditor::cut_graph_fragment(const std::vector<InputEdge>& inputs,
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
}

std::vector<std::string> onnx_import::ONNXModelEditor::model_inputs() const
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

std::string onnx_import::ONNXModelEditor::model_string() const
{
    return m_pimpl->m_model_proto.SerializeAsString();
}

void onnx_import::ONNXModelEditor::set_input_values(
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
