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

onnx_import::ONNXModelEditor::ONNXModelEditor(const std::string& model_path)
    : m_model_proto{new ONNX_NAMESPACE::ModelProto{}}
{
    onnx_import::parse_from_file(model_path, *m_model_proto);
}

onnx_import::ONNXModelEditor::~ONNXModelEditor() {
    delete m_model_proto;
}

void onnx_import::ONNXModelEditor::set_input_types(
    const std::map<std::string, element::Type_t>& input_types)
{
}
