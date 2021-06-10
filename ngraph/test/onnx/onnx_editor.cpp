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
using namespace onnx_editor;
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

    editor.cut_graph_fragment({{InputEdge(1, 0)}}, {});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_head_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_head_cut_ins_and_outs)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({{InputEdge(1, 0)}},
                              {{OutputEdge(2, 0)}});

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

    editor.cut_graph_fragment({{InputEdge(2, 0)}}, {});

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

    editor.cut_graph_fragment({}, {{OutputEdge{1, 0}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_tail_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__linear_model_tail_cut_ins_and_outs)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    editor.cut_graph_fragment({{InputEdge{0, 0}}}, {{OutputEdge{1, 0}}});

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

    editor.cut_graph_fragment({}, {{OutputEdge{1, 0}}});

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

    editor.cut_graph_fragment({}, {{OutputEdge{1, 0}}});

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

    editor.cut_graph_fragment({}, {{OutputEdge{0, 0}}});

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

    editor.cut_graph_fragment({{InputEdge{0, 2}}},
                              {{OutputEdge{0, 0}}});

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

    editor.cut_graph_fragment({{InputEdge{0, 2}}},
                              {{OutputEdge{0, 0}}});

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

    editor.cut_graph_fragment({}, {{OutputEdge{5, 1}}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__multiout_op_output_edge.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__existing_inputs_and_outputs_based_extraction)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{1, 1}, InputEdge{2, 0}}},
                              {{OutputEdge{4, 0}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__existing_inputs_and_outputs_based_extraction.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__twice_input_edge_from_tensor_with_single_consumer)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/add_ab.prototxt")};

    editor.cut_graph_fragment({InputEdge{1, 1}}, {});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__twice_input_edge_from_tensor_with_single_consumer.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{1, 0}, InputEdge{6, 0}}},
                              {{OutputEdge{6, 0}, OutputEdge{4, 0}}});

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

    editor.cut_graph_fragment({{InputEdge{3, 0}, InputEdge{3, 1}}},
                              {{OutputEdge{3, 0}, OutputEdge{4, 0}}});

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

    editor.cut_graph_fragment({{InputEdge{3, 0}, InputEdge{6, 0}}},
                              {{OutputEdge{6, 0}, OutputEdge{5, 1}}});

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

    editor.cut_graph_fragment({{InputEdge{1, 0}, InputEdge{3, 0}}}, {});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers_4.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_5)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({InputEdge{3, 0}},
                             {{OutputEdge{6,0}, OutputEdge{5, 1}}});

    // expected to behave the same way as the test above
    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers_5.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_custom_names)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    editor.cut_graph_fragment({{InputEdge{1, 0, "new_name_1"}, InputEdge{6, 0, "new_name_2"}}},
                              {{OutputEdge{6, 0}, OutputEdge{4, 0}}});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__input_edge_from_tensor_with_multiple_consumers_custom_names.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_input_relu2)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    editor.cut_graph_fragment({{InputEdge{4, 0}}}, {});

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

    editor.cut_graph_fragment({{InputEdge{2, 0}}}, {});

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

    editor.cut_graph_fragment({{InputEdge{2, 0}, InputEdge{3, 0}}}, {});

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

    editor.cut_graph_fragment({{InputEdge{5, 0}, InputEdge{3, 0}}}, {});

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
    try
    {
        editor.cut_graph_fragment({{InputEdge{15, 0}}}, {});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("The specified node index is out of range of nodes in the original model") !=
            std::string::npos);
    }
}

NGRAPH_TEST(onnx_editor, subgraph__invalid_port_idx)
{
    const auto model_path =
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt");

    ONNXModelEditor editor{model_path};
    try
    {
        editor.cut_graph_fragment({{InputEdge{0, 3}}}, {});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("The specified node with index: 0 has not input port with index: 3") !=
            std::string::npos);
    }
}

NGRAPH_TEST(onnx_editor, subgraph__inputs_getter)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    EXPECT_EQ(editor.model_inputs(),
              (std::vector<std::string>{"data_0", "conv1/7x7_s2_w_0", "conv1/7x7_s2_b_0"}));

    editor.cut_graph_fragment({{InputEdge{1, 0}}}, {});

    EXPECT_EQ(editor.model_inputs(), (std::vector<std::string>{"conv1/7x7_s2_1"}));
}

NGRAPH_TEST(onnx_editor, subgraph__custom_input_name_already_exist)
{
    const auto model_path =
        file_util::path_join(SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt");

    ONNXModelEditor editor{model_path};
    try
    {
        editor.cut_graph_fragment({{InputEdge{1, 0, "conv1/7x7_s2_b_0"}}}, {});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("New custom input name: conv1/7x7_s2_b_0 already exist in the graph") !=
            std::string::npos);
    }
}

// HIGHT LEVEL API TESTS
// INPUT EDGES TEST
NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_output_name_and_input_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    const InputEdge edge = editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_2"}},
                                                     EditorInput{"conv1/7x7_s2_1"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_1"}},
                                                      EditorInput{"data_0"});
    EXPECT_EQ(edge2.m_node_idx, 0);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_output_name_and_input_index)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    const InputEdge edge =
        editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_2"}}, EditorInput{0});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 =
        editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_1"}}, EditorInput{1});
    EXPECT_EQ(edge2.m_node_idx, 0);
    EXPECT_EQ(edge2.m_port_idx, 1);

    const InputEdge edge3 =
        editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_1"}}, EditorInput{2});
    EXPECT_EQ(edge3.m_node_idx, 0);
    EXPECT_EQ(edge3.m_port_idx, 2);
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_node_name_and_input_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    const InputEdge edge =
        editor.find_input_edge(EditorNode{"relu1"}, EditorInput{"conv1/7x7_s2_1"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 =
        editor.find_input_edge(EditorNode{"conv1"}, EditorInput{"conv1/7x7_s2_w_0"});
    EXPECT_EQ(edge2.m_node_idx, 0);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_node_name_and_input_index)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const InputEdge edge = editor.find_input_edge(EditorNode{"relu1_name"}, EditorInput{0});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"split_name"}, EditorInput{0});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_node_name_and_input_index_custom_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const InputEdge edge = editor.find_input_edge(EditorNode{"relu1_name"}, EditorInput{0, "custom_input_name_1"});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);
    EXPECT_EQ(edge.m_new_input_name, "custom_input_name_1");

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"split_name"}, EditorInput{0, "custom_input_name_2"});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 0);
    EXPECT_EQ(edge2.m_new_input_name, "custom_input_name_2");
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_empty_node_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    try
    {
        editor.find_input_edge(EditorNode{""}, EditorInput{"conv1/7x7_s2_1"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("Node with name: not_given and output_name: not_given was not found") !=
            std::string::npos);
    }
}

// OUTPUT EDGES TEST
NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_output_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const OutputEdge edge =
        editor.find_output_edge(EditorNode{EditorOutput{"mul2"}}, EditorOutput{"mul2"});
    EXPECT_EQ(edge.m_node_idx, 4);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 =
        editor.find_output_edge(EditorNode{EditorOutput{"split1"}}, EditorOutput{"split2"});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);

    // simplified overload
    const OutputEdge edge3 =
        editor.find_output_edge("mul2");
    EXPECT_EQ(edge3.m_node_idx, 4);
    EXPECT_EQ(edge3.m_port_idx, 0);

    const OutputEdge edge4 =
        editor.find_output_edge("split2");
    EXPECT_EQ(edge4.m_node_idx, 5);
    EXPECT_EQ(edge4.m_port_idx, 1);
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_output_name_and_output_index)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const OutputEdge edge =
        editor.find_output_edge(EditorNode{EditorOutput{"add2"}}, EditorOutput{0});
    EXPECT_EQ(edge.m_node_idx, 3);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 =
        editor.find_output_edge(EditorNode{EditorOutput{"split1"}}, EditorOutput{1});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);

    const OutputEdge edge3 =
        editor.find_output_edge(EditorNode{EditorOutput{"split2"}}, EditorOutput{0});
    EXPECT_EQ(edge3.m_node_idx, 5);
    EXPECT_EQ(edge3.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_node_name_and_output_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const OutputEdge edge =
        editor.find_output_edge(EditorNode{"relu1_name"}, EditorOutput{"relu1"});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 =
        editor.find_output_edge(EditorNode{"split_name"}, EditorOutput{"split2"});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_node_name_and_output_index)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"relu1_name"}, EditorOutput{0});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"split_name"}, EditorOutput{1});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

NGRAPH_TEST(onnx_editor, editor_api_select_edge_const_network)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    const InputEdge edge =
        editor.find_input_edge(EditorNode{EditorOutput{"relu4"}}, EditorInput{0});
    EXPECT_EQ(edge.m_node_idx, 3);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"relu4_name"}, EditorOutput{0});
    EXPECT_EQ(edge2.m_node_idx, 3);
    EXPECT_EQ(edge2.m_port_idx, 0);

    const OutputEdge edge3 = editor.find_output_edge(EditorNode{"add1_name"}, EditorOutput{0});
    EXPECT_EQ(edge3.m_node_idx, 4);
    EXPECT_EQ(edge3.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_select_edge_error_handling)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    // node with given output name not found
    try
    {
        editor.find_input_edge(EditorNode{EditorOutput{"not_existed"}}, EditorInput{0});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("Node with name: not_given and output_name: not_existed was not found") !=
            std::string::npos);
    }

    // node with given name not found
    try
    {
        editor.find_input_edge(EditorNode{"not_existed"}, EditorInput{0});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("Node with name: not_existed and output_name: not_given was not found") !=
            std::string::npos);
    }

    // input index out of scope
    try
    {
        editor.find_input_edge(EditorNode{"relu4_name"}, EditorInput{1});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not input with index: 1") !=
                    std::string::npos);
    }

    // output index out of scope
    try
    {
        editor.find_output_edge(EditorNode{"relu4_name"}, EditorOutput{1});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not output with index: 1") !=
                    std::string::npos);
    }

    // input name not found
    try
    {
        editor.find_input_edge(EditorNode{"relu4_name"}, EditorInput{"not_existed"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not input with name: not_existed") !=
                    std::string::npos);
    }

    // output name not found
    try
    {
        editor.find_output_edge(EditorNode{"relu4_name"}, EditorOutput{"not_existed"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not output with name: not_existed") !=
                    std::string::npos);
    }
}

// Nodes with ambiguous node names tests
NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_ambiguous_node_name_but_matched_input)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    InputEdge edge = editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"in2"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 1);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"add1"});
    EXPECT_EQ(edge2.m_node_idx, 3);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_ambiguous_node_name_and_not_matched_input)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    try
    {
        editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"in3"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Input edge described by: add_ambiguous_name and input name: in3 was not found") !=
                    std::string::npos);
    }

    try
    {
        editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"relu1"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Given node name: add_ambiguous_name and input name: relu1 are ambiguous to determine input edge") !=
                    std::string::npos);
    }
}

NGRAPH_TEST(onnx_editor, editor_api_select_input_edge_by_ambiguous_node_name_and_input_index)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    try
    {
        editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{0});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Given node name: add_ambiguous_name and input index: 0 are ambiguous to determine input edge") !=
                    std::string::npos);
    }
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_ambiguous_node_name_but_matched_output)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{"add1"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{"add2"});
    EXPECT_EQ(edge2.m_node_idx, 3);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_the_same_node_name_and_output_name)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests_2.prototxt")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"add1"}, EditorOutput{0});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{EditorOutput{"add1"}}, EditorOutput{0});
    EXPECT_EQ(edge2.m_node_idx, 4);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_ambiguous_node_name_and_not_matched_output)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    try
    {
        editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{"split2"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Output edge described by: add_ambiguous_name and output name: split2 was not found") !=
                    std::string::npos);
    }
}

NGRAPH_TEST(onnx_editor, editor_api_select_output_edge_by_ambiguous_node_name_and_output_index)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    try
    {
        editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{0});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Given node name: add_ambiguous_name and output index: 0 are ambiguous to determine output edge") !=
                    std::string::npos);
    }
}

NGRAPH_TEST(onnx_editor, editor_api_use_edge_mapper_with_graph_cutter)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    // InputEdge{1, "in2"}
    const auto input_edge_1 = editor.find_input_edge(
                                   EditorNode(EditorOutput("add1")), EditorInput(1));
    // InputEdge{2, "in3"}
    const auto input_edge_2 = editor.find_input_edge(
                                   EditorNode(EditorOutput("conv1")), EditorInput(0));


    const auto output_edge = editor.find_output_edge(
                                  EditorNode(EditorOutput("mul2")), EditorOutput(0));
    // OutputEdge{4, "mul2"}
    editor.cut_graph_fragment({input_edge_1, input_edge_2}, {output_edge});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__existing_inputs_and_outputs_based_extraction.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;

    // check if mapper was updated after the model changed
    const auto input_edge_4 = editor.find_input_edge(
                                EditorNode(EditorOutput("relu1")), EditorInput(0));
    EXPECT_EQ(input_edge_4.m_node_idx, 0);
    EXPECT_EQ(input_edge_4.m_port_idx, 0);

    const auto input_edge_5 = editor.find_input_edge(
                                EditorNode(EditorOutput("add1")), EditorInput(1));
    EXPECT_EQ(input_edge_5.m_node_idx, 1);
    EXPECT_EQ(input_edge_5.m_port_idx, 1);

    const auto output_edge_3 = editor.find_output_edge("mul2");
    EXPECT_EQ(output_edge_3.m_node_idx, 3);
    EXPECT_EQ(output_edge_3.m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_use_edge_mapper_with_graph_cutter_custom_names)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const auto input_edge_1 = editor.find_input_edge(
                                   EditorNode{EditorOutput{"mul2"}}, EditorInput{1, "new_name_1"});
    const auto input_edge_2 = editor.find_input_edge(
                                   EditorNode{EditorOutput{"split2"}}, EditorInput{"add2", "new_name_2"});

    editor.cut_graph_fragment({input_edge_1, input_edge_2}, {});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__use_edge_mapper_with_graph_cutter_custom_names.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, editor_api_find_output_consumers)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    std::vector<InputEdge> output_consumers = editor.find_output_consumers("relu1");
    EXPECT_EQ(output_consumers.size(), 3);
    EXPECT_EQ(output_consumers[0].m_node_idx, 1);
    EXPECT_EQ(output_consumers[0].m_port_idx, 0);
    EXPECT_EQ(output_consumers[1].m_node_idx, 3);
    EXPECT_EQ(output_consumers[1].m_port_idx, 0);
    EXPECT_EQ(output_consumers[2].m_node_idx, 6);
    EXPECT_EQ(output_consumers[2].m_port_idx, 0);

    output_consumers = editor.find_output_consumers("add1");
    EXPECT_EQ(output_consumers.size(), 2);
    EXPECT_EQ(output_consumers[0].m_node_idx, 3);
    EXPECT_EQ(output_consumers[0].m_port_idx, 1);
    EXPECT_EQ(output_consumers[1].m_node_idx, 4);
    EXPECT_EQ(output_consumers[1].m_port_idx, 0);

    output_consumers = editor.find_output_consumers("in3");
    EXPECT_EQ(output_consumers.size(), 1);
    EXPECT_EQ(output_consumers[0].m_node_idx, 2);
    EXPECT_EQ(output_consumers[0].m_port_idx, 0);
}

NGRAPH_TEST(onnx_editor, editor_api_find_output_consumers_empty_result)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    const std::vector<InputEdge> output_consumers = editor.find_output_consumers("not_existed");
    EXPECT_EQ(output_consumers.size(), 0);
}

NGRAPH_TEST(onnx_editor, editor_api_is_correct_and_unambiguous_node)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph_extraction_tests.prototxt")};

    bool is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"relu1"}});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"mul2"}});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"split2"}});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{"relu1_name"});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"in3"}});
    EXPECT_EQ(is_correct_node, false);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{"add_ambiguous_name"});
    EXPECT_EQ(is_correct_node, false);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{"not_exist"});
    EXPECT_EQ(is_correct_node, false);
}

NGRAPH_TEST(onnx_editor, editor_api_input_edge_from_tensor_with_single_consumer)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/add_ab.prototxt")};

    const auto edge = editor.find_input_edge(EditorNode{EditorOutput{"Y"}}, EditorInput{1});
    editor.cut_graph_fragment({edge}, {});

    const auto ref_model =
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/model_editor/reference/"
                             "subgraph__twice_input_edge_from_tensor_with_single_consumer.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

NGRAPH_TEST(onnx_editor, editor_api_input_edge_from_tensor_with_single_consumer_ambiguous)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/add_ab.prototxt")};

    try
    {
        editor.find_input_edge(EditorNode{EditorOutput{"Y"}}, EditorInput{"X"});
    }
    catch (const std::exception& e)
    {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 1 has more than one inputs with name: X") !=
                    std::string::npos);
    }
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

NGRAPH_TEST(onnx_editor, combined__cut_and_replace_shape)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/subgraph__inception_head.prototxt")};

    const auto new_shape = PartialShape({1, 64, 112, 112});
    editor.cut_graph_fragment({{InputEdge(1, 0)}}, {});
    editor.set_input_shapes({{"conv1/7x7_s2_1", new_shape}});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/subgraph__linear_model_head_cut.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;

    const auto graph_inputs = editor.get_function()->get_parameters();
    EXPECT_TRUE(
        find_input(graph_inputs, "conv1/7x7_s2_1")->get_partial_shape().same_scheme(new_shape));
}

NGRAPH_TEST(onnx_editor, cut_operator_with_no_schema)
{
    ONNXModelEditor editor{file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/unknown_input_value_info.prototxt")};

    editor.cut_graph_fragment({{InputEdge{1, 0}}}, {});

    const auto ref_model = file_util::path_join(
        SERIALIZED_ZOO, "onnx/model_editor/reference/unknown_input_value_info.prototxt");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}
