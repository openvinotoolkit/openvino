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

#include <algorithm>
#include "gtest/gtest.h"

#include "default_opset.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_import/editor/editor.hpp"
#include "onnx_import/onnx.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

namespace
{
    using InputTypePred = std::function<bool(const std::shared_ptr<ngraph::Node>)>;

    // A higher order factory function that produces predicates bound to a particular element type
    InputTypePred element_type_is(const element::Type et)
    {
        return [et](const std::shared_ptr<ngraph::Node> input) {
            return input->get_element_type() == et;
        };
    }
} // namespace

NGRAPH_TEST(onnx_editor, single_input_type_substitution)
{
    // the original model contains 2 inputs with i64 data type and one f32 input
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    editor.set_input_types({{"A", element::i64}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    const auto float_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::f32));

    const auto integer_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i64));

    EXPECT_EQ(float_inputs_count, 0);
    EXPECT_EQ(integer_inputs_count, 3);

    const auto input_a = std::find_if(
        std::begin(graph_inputs),
        std::end(graph_inputs),
        [](const std::shared_ptr<op::Parameter> i) { return i->get_friendly_name() == "A"; });

    ASSERT_NE(input_a, std::end(graph_inputs));
    EXPECT_EQ(input_a->get()->get_element_type(), element::i64);
}

NGRAPH_TEST(onnx_editor, all_inputs_type_substitution)
{
    // the original model contains 2 inputs with i64 data type and one f32 input
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    editor.set_input_types({{"A", element::i8}, {"B", element::i8}, {"C", element::i8}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    const auto float_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::f32));

    const auto integer_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i8));

    EXPECT_EQ(float_inputs_count, 0);
    EXPECT_EQ(integer_inputs_count, 3);
}

NGRAPH_TEST(onnx_editor, missing_type_in_input_descriptor)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/invalid_input_no_type.prototxt")};

    // input A doesn't have the "type" field in the model and so the data type cannot be modified
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, missing_tensor_type_in_input_descriptor)
{
    onnx_import::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/invalid_input_no_tensor_type.prototxt")};

    // input A doesn't have the "tensor_type" field in the model
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, unsupported_data_type_passed)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    EXPECT_THROW(editor.set_input_types({{"A", element::dynamic}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, incorrect_input_name_passed)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    EXPECT_THROW(editor.set_input_types({{"ShiaLaBeouf", element::i64}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, elem_type_missing_in_input)
{
    // the original model contains 2 inputs with i64 data type and one f32 input
    onnx_import::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/elem_type_missing_in_input.prototxt")};

    // the "elem_type" is missing in the model but it should be possible to set the type anyway
    EXPECT_NO_THROW(editor.set_input_types({{"A", element::i64}}));

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    const auto integer_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i64));

    EXPECT_EQ(integer_inputs_count, 2);

    const auto function_result = function->get_result();
    EXPECT_EQ(function_result->get_element_type(), element::i64);
}
