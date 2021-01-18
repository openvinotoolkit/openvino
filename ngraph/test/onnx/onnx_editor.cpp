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

    std::shared_ptr<op::Parameter> find_input(const ParameterVector& inputs,
                                              const std::string& name)
    {
        const auto input_pos = std::find_if(
            std::begin(inputs), std::end(inputs), [&name](const ParameterVector::value_type i) {
                return i->get_friendly_name() == name;
            });

        return *input_pos;
    }
} // namespace

NGRAPH_TEST(onnx_editor, types__single_input_type_substitution)
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

    EXPECT_EQ(find_input(graph_inputs, "A")->get_element_type(), element::i64);
}

NGRAPH_TEST(onnx_editor, types__all_inputs_type_substitution)
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

NGRAPH_TEST(onnx_editor, types__missing_type_in_input_descriptor)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/invalid_input_no_type.prototxt")};

    // input A doesn't have the "type" field in the model and so the data type cannot be modified
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__missing_tensor_type_in_input_descriptor)
{
    onnx_import::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/invalid_input_no_tensor_type.prototxt")};

    // input A doesn't have the "tensor_type" field in the model
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__unsupported_data_type_passed)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    EXPECT_THROW(editor.set_input_types({{"A", element::dynamic}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__incorrect_input_name_passed)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    EXPECT_THROW(editor.set_input_types({{"ShiaLaBeouf", element::i64}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__elem_type_missing_in_input)
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

NGRAPH_TEST(onnx_editor, shapes__modify_single_input)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{1};

    editor.set_input_shapes({{"B", new_shape}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "B")->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, shapes__modify_all_inputs)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{1, 2, 3, 5, 8, 13};

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    for (const auto& input : graph_inputs)
    {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

NGRAPH_TEST(onnx_editor, shapes__dynamic_rank_in_model)
{
    onnx_import::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/shapes__dynamic_rank_in_model.prototxt")};

    // input A in the model doesn't have the "shape" field meaning it has dynamic rank
    // it should still be possible to set such input's shape to some custom value
    const auto expected_shape_of_A = PartialShape{1, 2};
    EXPECT_NO_THROW(editor.set_input_shapes({{"A", expected_shape_of_A}}));

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(
        find_input(graph_inputs, "A")->get_partial_shape().same_scheme(expected_shape_of_A));
}

NGRAPH_TEST(onnx_editor, shapes__set_dynamic_dimension)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{Dimension::dynamic()};

    editor.set_input_shapes({{"A", new_shape}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "A")->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, shapes__set_mixed_dimensions)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape_A = PartialShape{21, Dimension::dynamic()};
    const auto new_shape_B = PartialShape{Dimension::dynamic(), 37};

    editor.set_input_shapes({{"A", new_shape_A}, {"B", new_shape_B}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape_A));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape_B));
}

NGRAPH_TEST(onnx_editor, shapes__set_scalar_inputs)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{};

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, shapes__static_to_dynamic_rank_substitution)
{
    onnx_import::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape::dynamic();

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = onnx_import::import_onnx_model(editor);

    const auto graph_inputs = function->get_parameters();

    for (const auto& input : graph_inputs)
    {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}
