// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>

#include "gtest/gtest.h"

#include "default_opset.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "onnx_editor/editor.hpp"
#include "onnx_import/onnx.hpp"
#include "util/engine/interpreter_engine.hpp"
#include "util/onnx_test_util.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;
using namespace ngraph::onnx_import;
using namespace ngraph::onnx_editor;
using namespace ngraph::test;

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
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    editor.set_input_types({{"A", element::i64}});

    const auto function = editor.get_function();
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
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    editor.set_input_types({{"A", element::i8}, {"B", element::i8}, {"C", element::i8}});

    const auto function = editor.get_function();

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
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/invalid_input_no_type.prototxt")};

    // input A doesn't have the "type" field in the model and so the data type cannot be modified
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__missing_tensor_type_in_input_descriptor)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/invalid_input_no_tensor_type.prototxt")};

    // input A doesn't have the "tensor_type" field in the model
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__unsupported_data_type_passed)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    EXPECT_THROW(editor.set_input_types({{"A", element::dynamic}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__incorrect_input_name_passed)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_abc.prototxt")};

    EXPECT_THROW(editor.set_input_types({{"ShiaLaBeouf", element::i64}}), ngraph_error);
}

NGRAPH_TEST(onnx_editor, types__elem_type_missing_in_input)
{
    // the original model contains 2 inputs with i64 data type and one f32 input
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/elem_type_missing_in_input.prototxt")};

    // the "elem_type" is missing in the model but it should be possible to set the type anyway
    EXPECT_NO_THROW(editor.set_input_types({{"A", element::i64}}));

    const auto function = editor.get_function();

    const auto graph_inputs = function->get_parameters();

    const auto integer_inputs_count = std::count_if(
        std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i64));

    EXPECT_EQ(integer_inputs_count, 2);

    const auto function_result = function->get_result();
    EXPECT_EQ(function_result->get_element_type(), element::i64);
}

NGRAPH_TEST(onnx_editor, shapes__modify_single_input)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{1};

    editor.set_input_shapes({{"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "B")->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, shapes__modify_all_inputs)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{1, 2, 3, 5, 8, 13};

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    for (const auto& input : graph_inputs)
    {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

NGRAPH_TEST(onnx_editor, shapes__dynamic_rank_in_model)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/shapes__dynamic_rank_in_model.prototxt")};

    // input A in the model doesn't have the "shape" field meaning it has dynamic rank
    // it should still be possible to set such input's shape to some custom value
    const auto expected_shape_of_A = PartialShape{1, 2};
    EXPECT_NO_THROW(editor.set_input_shapes({{"A", expected_shape_of_A}}));

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(
        find_input(graph_inputs, "A")->get_partial_shape().same_scheme(expected_shape_of_A));
}

NGRAPH_TEST(onnx_editor, shapes__set_dynamic_dimension)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{Dimension::dynamic()};

    editor.set_input_shapes({{"A", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "A")->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, shapes__set_mixed_dimensions)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape_A = PartialShape{21, Dimension::dynamic()};
    const auto new_shape_B = PartialShape{Dimension::dynamic(), 37};

    editor.set_input_shapes({{"A", new_shape_A}, {"B", new_shape_B}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape_A));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape_B));
}

NGRAPH_TEST(onnx_editor, shapes__set_scalar_inputs)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape{};

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, shapes__static_to_dynamic_rank_substitution)
{
    ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/shapes__add_two_inputs.prototxt")};

    const auto new_shape = PartialShape::dynamic();

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    for (const auto& input : graph_inputs)
    {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_head_cut)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({{InputEdge(1, "conv1/7x7_s2_1")}}, {});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_head_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_head_cut_ins_and_outs)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({{InputEdge(1, "conv1/7x7_s2_1")}},
                              {{OutputEdge(2, "pool1/3x3_s2_1")}});

    // expected to behave the same way as subgraph__linear_model_head_cut
    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_head_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_deeper_head_cut)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({{InputEdge(2, "conv1/7x7_s2_2")}}, {});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__linear_model_deeper_head_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_tail_cut)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({}, {{OutputEdge{1, "conv1/7x7_s2_2"}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_tail_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_tail_cut_ins_and_outs)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({{InputEdge{0, "data_0"}}}, {{OutputEdge{1, "conv1/7x7_s2_2"}}});

    // expected to behave the same way as subgraph__linear_model_tail_cut
    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_tail_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_with_initializer_tail_cut)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head_with_initializer.prototxt")};

    editor.cut_graph_fragment({}, {{OutputEdge{1, "conv1/7x7_s2_2"}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__linear_model_with_initializer_tail_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__initializer_without_matching_input_tail_cut)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__initializer_without_matching_input.prototxt")};

    editor.cut_graph_fragment({}, {{OutputEdge{1, "conv1/7x7_s2_2"}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__initializer_without_matching_input_tail_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_deeper_tail_cut)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({}, {{OutputEdge{0, "conv1/7x7_s2_1"}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__linear_model_deeper_tail_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__no_input_params)
{
    const auto model_path =
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt");

    ONNXModelEditor editor{model_path};

    editor.cut_graph_fragment({}, {});

    const auto result = compare_onnx_models(editor.model_string(), model_path);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__initializer_to_input_replacement)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head_with_initializer.prototxt")};

    editor.cut_graph_fragment({{InputEdge{0, "conv1/7x7_s2_b_0"}}},
                              {{OutputEdge{0, "conv1/7x7_s2_1"}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__initializer_to_input_replacement.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__initializer_to_input_replacement_2)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__initializer_without_matching_input.prototxt")};

    editor.cut_graph_fragment({{InputEdge{0, "conv1/7x7_s2_b_0"}}},
                              {{OutputEdge{0, "conv1/7x7_s2_1"}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__initializer_to_input_replacement.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__multiout_op_output_edge)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({}, {{OutputEdge{5, "split2"}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__multiout_op_output_edge.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__existing_inputs_and_outputs_based_extraction)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{1, "in2"}, InputEdge{2, "in3"}}},
                              {{OutputEdge{4, "mul2"}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__existing_inputs_and_outputs_based_extraction.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{1, "relu1"}, InputEdge{6, "relu1"}}},
                              {{OutputEdge{6, "mul1"}, OutputEdge{4, "mul2"}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_2)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{3, "relu1"}, InputEdge{3, "add1"}}},
                              {{OutputEdge{3, "add2"}, OutputEdge{4, "mul2"}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers_2.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_3)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{3, "relu1"}, InputEdge{6, "relu1"}}},
                              {{OutputEdge{6, "mul1"}, OutputEdge{5, "split2"}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers_3.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_4)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{3, "relu1"}}},
                              {{OutputEdge{6, "mul1"}, OutputEdge{5, "split2"}}});

    // expected to behave the same way as the test above
    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers_3.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_input_relu2)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    editor.cut_graph_fragment({{InputEdge{4, "relu2"}}}, {});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__multiple_consumers_of_graph_input_relu2.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    editor.cut_graph_fragment({{InputEdge{2, "in2"}}}, {});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__multiple_consumers_of_graph_initializer.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer_2)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    editor.cut_graph_fragment({{InputEdge{2, "in2"}, InputEdge{3, "in2"}}}, {});

    // same as above
    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__multiple_consumers_of_graph_initializer.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer_relu2_and_init)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    editor.cut_graph_fragment({{InputEdge{5, "relu2"}, InputEdge{3, "in2"}}}, {});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/"
        "subgraph__multiple_consumers_of_graph_initializer_relu2_and_init.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__invalid_edge_idx)
{
    const auto model_path =
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt");

    ONNXModelEditor editor{model_path};

    EXPECT_THROW(editor.cut_graph_fragment({{InputEdge{15, "x"}}}, {}), ngraph::ngraph_error);
}

NGRAPH_TEST(onnx_editor, subgraph__invalid_edge_name)
{
    const auto model_path =
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt");

    ONNXModelEditor editor{model_path};

    EXPECT_THROW(editor.cut_graph_fragment({{InputEdge{0, "x"}}}, {}), ngraph::ngraph_error);
}

NGRAPH_TEST(onnx_editor, subgraph__inputs_getter)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    EXPECT_EQ(editor.model_inputs(),
              (std::vector<std::string>{"data_0", "conv1/7x7_s2_w_0", "conv1/7x7_s2_b_0"}));

    editor.cut_graph_fragment({{InputEdge(1, "conv1/7x7_s2_1")}}, {});

    EXPECT_EQ(editor.model_inputs(), (std::vector<std::string>{"conv1/7x7_s2_2:conv1/7x7_s2_1"}));
}

using TestEngine = test::INTERPRETER_Engine;

NGRAPH_TEST(onnx_editor, values__append_one_initializer)
{
    onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_1D.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", op::Constant::create(element::i64, Shape{2}, {1, 2}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int64_t>(Shape{2}, {5, 6});
    test_case.add_expected_output<int64_t>(Shape{2}, {6, 8});
    test_case.run();
}

NGRAPH_TEST(onnx_editor, values__append_two_initializers_to_invalid)
{
    onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_1D_invalid.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", op::Constant::create(element::i64, Shape{2}, {4, 2}));
    in_vals.emplace("B", op::Constant::create(element::i64, Shape{2}, {1, 3}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 5});
    test_case.run();
}

NGRAPH_TEST(onnx_editor, values__modify_one_initializer)
{
    onnx_editor::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/add_1D_with_initializers.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("B", op::Constant::create(element::i64, Shape{2}, {3, 4}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {4, 6});
    test_case.run();
}

NGRAPH_TEST(onnx_editor, values__modify_two_initializers)
{
    onnx_editor::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/add_1D_with_initializers.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", op::Constant::create(element::i64, Shape{2}, {3, 6}));
    in_vals.emplace("B", op::Constant::create(element::i64, Shape{2}, {2, 1}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 7});
    test_case.run();
}

NGRAPH_TEST(onnx_editor, values__no_inputs_modify_two_initializers)
{
    onnx_editor::ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/add_1D_with_initializers_only.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", op::Constant::create(element::i64, Shape{2}, {1, 2}));
    in_vals.emplace("B", op::Constant::create(element::i64, Shape{2}, {11, 22}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {12, 24});
    test_case.run();
}

NGRAPH_TEST(onnx_editor, values__append_two_initializers_change_shape_type)
{
    onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/add_1D.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", op::Constant::create(element::i8, Shape{2, 1}, {-1, 1}));
    in_vals.emplace("B", op::Constant::create(element::i8, Shape{2, 1}, {-2, 2}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<int8_t>(Shape{2, 1}, {-3, 3});
    test_case.run();
}

NGRAPH_TEST(onnx_editor, values__append_two_initializers_mixed_types)
{
    onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_elements_float_3D_axis_2.prototxt")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("data",
                    op::Constant::create(element::i16, Shape{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    in_vals.emplace("indices", op::Constant::create(element::i32, Shape{2, 2, 1}, {0, 1, 0, 1}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_expected_output<int16_t>(Shape{2, 2, 1}, {1, 4, 5, 8});
    test_case.run();
}
