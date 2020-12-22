//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <onnx/onnx_pb.h>

#include "onnx_import/editor/editor.hpp"
#include "onnx_import/utils/parser.hpp"

using namespace ngraph;

namespace
{
    ONNX_NAMESPACE::ValueInfoProto* find_graph_input(ONNX_NAMESPACE::GraphProto& graph,
                                                     const std::string& name)
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

    void modify_input_type(ONNX_NAMESPACE::ValueInfoProto& onnx_input, element::Type_t elem_type)
    {
        const std::string malformed_input_error_msg =
            "The input is malformed, cannot change the data type. Input name: " + onnx_input.name();

        if (!onnx_input.has_type())
        {
            throw ngraph_error(malformed_input_error_msg);
        }
        else
        {
            auto* type_proto = onnx_input.mutable_type();
            if (!type_proto->has_tensor_type())
            {
                throw ngraph_error(malformed_input_error_msg);
            }
            else
            {
                auto* tensor_type = type_proto->mutable_tensor_type();
                tensor_type->set_elem_type(7);
            }
        }
    }
} // namespace

onnx_import::ONNXModelEditor::ONNXModelEditor(const std::string& model_path)
    : m_model_proto{new ONNX_NAMESPACE::ModelProto{}}
{
    onnx_import::parse_from_file(model_path, *m_model_proto);
}

onnx_import::ONNXModelEditor::~ONNXModelEditor()
{
    delete m_model_proto;
}

void onnx_import::ONNXModelEditor::set_input_types(
    const std::map<std::string, element::Type_t>& input_types)
{
    auto* onnx_graph = m_model_proto->mutable_graph();

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
