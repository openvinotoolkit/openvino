// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "default_opset.hpp"
#include "editor.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_test_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ov;
using namespace ov::onnx_editor;
using namespace ngraph::test;

static std::string s_manifest = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(), "${MANIFEST}");

namespace {
using InputTypePred = std::function<bool(const std::shared_ptr<ngraph::Node>)>;

// A higher order factory function that produces predicates bound to a particular element type
InputTypePred element_type_is(const element::Type et) {
    return [et](const std::shared_ptr<ngraph::Node> input) {
        return input->get_element_type() == et;
    };
}

std::shared_ptr<op::v0::Parameter> find_input(const ParameterVector& inputs, const std::string& name) {
    const auto input_pos =
        std::find_if(std::begin(inputs), std::end(inputs), [&name](const ParameterVector::value_type i) {
            return i->get_friendly_name() == name;
        });

    return *input_pos;
}
}  // namespace

OPENVINO_TEST(onnx_editor, types__single_input_type_substitution) {
    // the original model contains 2 inputs with i64 data type and one f32 input
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_abc.onnx")};

    editor.set_input_types({{"A", element::i64}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    const auto float_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::f32));

    const auto integer_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i64));

    EXPECT_EQ(float_inputs_count, 0);
    EXPECT_EQ(integer_inputs_count, 3);

    EXPECT_EQ(find_input(graph_inputs, "A")->get_element_type(), element::i64);
}

OPENVINO_TEST(onnx_editor, types__all_inputs_type_substitution) {
    // the original model contains 2 inputs with i64 data type and one f32 input
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_abc.onnx")};

    editor.set_input_types({{"A", element::i8}, {"B", element::i8}, {"C", element::i8}});

    const auto function = editor.get_function();

    const auto graph_inputs = function->get_parameters();

    const auto float_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::f32));

    const auto integer_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i8));

    EXPECT_EQ(float_inputs_count, 0);
    EXPECT_EQ(integer_inputs_count, 3);
}

OPENVINO_TEST(onnx_editor, types__missing_type_in_input_descriptor) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/invalid_input_no_type.onnx")};

    // input A doesn't have the "type" field in the model and so the data type cannot be modified
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph::ngraph_error);
}

OPENVINO_TEST(onnx_editor, types__missing_tensor_type_in_input_descriptor) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/invalid_input_no_tensor_type.onnx")};

    // input A doesn't have the "tensor_type" field in the model
    EXPECT_THROW(editor.set_input_types({{"A", element::f32}}), ngraph::ngraph_error);
}

OPENVINO_TEST(onnx_editor, types__unsupported_data_type_passed) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_abc.onnx")};

    EXPECT_THROW(editor.set_input_types({{"A", element::dynamic}}), ngraph::ngraph_error);
}

OPENVINO_TEST(onnx_editor, types__incorrect_input_name_passed) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_abc.onnx")};

    EXPECT_THROW(editor.set_input_types({{"ShiaLaBeouf", element::i64}}), ngraph::ngraph_error);
}

OPENVINO_TEST(onnx_editor, types__elem_type_missing_in_input) {
    // the original model contains 2 inputs with i64 data type and one f32 input
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/elem_type_missing_in_input.onnx")};

    // the "elem_type" is missing in the model but it should be possible to set the type anyway
    EXPECT_NO_THROW(editor.set_input_types({{"A", element::i64}}));

    const auto function = editor.get_function();

    const auto graph_inputs = function->get_parameters();

    const auto integer_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(element::i64));

    EXPECT_EQ(integer_inputs_count, 2);

    const auto function_result = function->get_result();
    EXPECT_EQ(function_result->get_element_type(), element::i64);
}

OPENVINO_TEST(onnx_editor, shapes__modify_single_input) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__add_two_inputs.onnx")};

    const auto new_shape = PartialShape{1};

    editor.set_input_shapes({{"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "B")->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, shapes__modify_all_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__add_two_inputs.onnx")};

    const auto new_shape = PartialShape{1, 2, 3, 5, 8, 13};

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    for (const auto& input : graph_inputs) {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

OPENVINO_TEST(onnx_editor, shapes__dynamic_rank_in_model) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__dynamic_rank_in_model.onnx")};

    // input A in the model doesn't have the "shape" field meaning it has dynamic rank
    // it should still be possible to set such input's shape to some custom value
    const auto expected_shape_of_A = PartialShape{1, 2};
    EXPECT_NO_THROW(editor.set_input_shapes({{"A", expected_shape_of_A}}));

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "A")->get_partial_shape().same_scheme(expected_shape_of_A));
}

OPENVINO_TEST(onnx_editor, shapes__set_dynamic_dimension) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__add_two_inputs.onnx")};

    const auto new_shape = PartialShape{Dimension::dynamic()};

    editor.set_input_shapes({{"A", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "A")->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, shapes__set_mixed_dimensions) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__add_two_inputs.onnx")};

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

OPENVINO_TEST(onnx_editor, shapes__set_scalar_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__add_two_inputs.onnx")};

    const auto new_shape = PartialShape{};

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, shapes__static_to_dynamic_rank_substitution) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/shapes__add_two_inputs.onnx")};

    const auto new_shape = PartialShape::dynamic();

    editor.set_input_shapes({{"A", new_shape}, {"B", new_shape}});

    const auto function = editor.get_function();
    const auto graph_inputs = function->get_parameters();

    for (const auto& input : graph_inputs) {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_head_cut) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    editor.extract_subgraph({{InputEdge(1, 0)}}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_head_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_head_cut_ins_and_outs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    editor.extract_subgraph({{InputEdge(1, 0)}}, {{OutputEdge(2, 0)}});

    // expected to behave the same way as subgraph__linear_model_head_cut
    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_head_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_deeper_head_cut) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    editor.extract_subgraph({{InputEdge(2, 0)}}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_deeper_head_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_tail_cut) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    editor.extract_subgraph({}, {{OutputEdge{1, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_tail_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_tail_cut_ins_and_outs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    editor.extract_subgraph({{InputEdge{0, 0}}}, {{OutputEdge{1, 0}}});

    // expected to behave the same way as subgraph__linear_model_tail_cut
    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_tail_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_with_initializer_tail_cut) {
    ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/subgraph__inception_head_with_initializer.onnx")};

    editor.extract_subgraph({}, {{OutputEdge{1, 0}}});

    const auto ref_model = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__linear_model_with_initializer_tail_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__initializer_without_matching_input_tail_cut) {
    ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/subgraph__initializer_without_matching_input.onnx")};

    editor.extract_subgraph({}, {{OutputEdge{1, 0}}});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/"
                                                        "subgraph__initializer_without_matching_input_tail_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_deeper_tail_cut) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    editor.extract_subgraph({}, {{OutputEdge{0, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_deeper_tail_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__no_input_params) {
    const auto model_path = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                         SERIALIZED_ZOO,
                                                         "onnx/model_editor/subgraph__inception_head.onnx");

    ONNXModelEditor editor{model_path};

    editor.extract_subgraph({}, {});

    const auto result = compare_onnx_models(editor.model_string(), model_path);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__initializer_to_input_replacement) {
    ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/subgraph__inception_head_with_initializer.onnx")};

    editor.extract_subgraph({{InputEdge{0, 2}}}, {{OutputEdge{0, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__initializer_to_input_replacement.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__initializer_to_input_replacement_2) {
    ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/subgraph__initializer_without_matching_input.onnx")};

    editor.extract_subgraph({{InputEdge{0, 2}}}, {{OutputEdge{0, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__initializer_to_input_replacement.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiout_op_output_edge) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({}, {{OutputEdge{5, 1}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__multiout_op_output_edge.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__existing_inputs_and_outputs_based_extraction) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({{InputEdge{1, 1}, InputEdge{2, 0}}}, {{OutputEdge{4, 0}}});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/"
                                                        "subgraph__existing_inputs_and_outputs_based_extraction.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__twice_input_edge_from_tensor_with_single_consumer) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab.onnx")};

    editor.extract_subgraph({InputEdge{1, 1}}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__twice_input_edge_from_tensor_with_single_consumer.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({{InputEdge{1, 0}, InputEdge{6, 0}}}, {{OutputEdge{6, 0}, OutputEdge{4, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__input_edge_from_tensor_with_multiple_consumers.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_2) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({{InputEdge{3, 0}, InputEdge{3, 1}}}, {{OutputEdge{3, 0}, OutputEdge{4, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__input_edge_from_tensor_with_multiple_consumers_2.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_3) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({{InputEdge{3, 0}, InputEdge{6, 0}}}, {{OutputEdge{6, 0}, OutputEdge{5, 1}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__input_edge_from_tensor_with_multiple_consumers_3.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_4) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({{InputEdge{1, 0}, InputEdge{3, 0}}}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__input_edge_from_tensor_with_multiple_consumers_4.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_5) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({InputEdge{3, 0}}, {{OutputEdge{6, 0}, OutputEdge{5, 1}}});

    // expected to behave the same way as the test above
    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__input_edge_from_tensor_with_multiple_consumers_5.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_custom_names) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    editor.extract_subgraph({{InputEdge{1, 0, "new_name_1"}, InputEdge{6, 0, "new_name_2"}}},
                            {{OutputEdge{6, 0}, OutputEdge{4, 0}}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__input_edge_from_tensor_with_multiple_consumers_custom_names.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_input_relu2) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    editor.extract_subgraph({{InputEdge{4, 0}}}, {});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/"
                                                        "subgraph__multiple_consumers_of_graph_input_relu2.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    editor.extract_subgraph({{InputEdge{2, 0}}}, {});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/"
                                                        "subgraph__multiple_consumers_of_graph_initializer.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer_2) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    editor.extract_subgraph({{InputEdge{2, 0}, InputEdge{3, 0}}}, {});

    // same as above
    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/"
                                                        "subgraph__multiple_consumers_of_graph_initializer.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer_relu2_and_init) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    editor.extract_subgraph({{InputEdge{5, 0}, InputEdge{3, 0}}}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__multiple_consumers_of_graph_initializer_relu2_and_init.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__invalid_edge_idx) {
    const auto model_path = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                         SERIALIZED_ZOO,
                                                         "onnx/model_editor/subgraph__inception_head.onnx");

    ONNXModelEditor editor{model_path};
    try {
        editor.extract_subgraph({{InputEdge{15, 0}}}, {});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("The specified node index is out of range of nodes in the original model") !=
                    std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, subgraph__invalid_port_idx) {
    const auto model_path = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                         SERIALIZED_ZOO,
                                                         "onnx/model_editor/subgraph__inception_head.onnx");

    ONNXModelEditor editor{model_path};
    try {
        editor.extract_subgraph({{InputEdge{0, 3}}}, {});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("The specified node with index: 0 has not input port with index: 3") != std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, subgraph__inputs_getter) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    EXPECT_EQ(editor.model_inputs(), (std::vector<std::string>{"data_0", "conv1/7x7_s2_w_0", "conv1/7x7_s2_b_0"}));

    editor.extract_subgraph({{InputEdge{1, 0}}}, {});

    EXPECT_EQ(editor.model_inputs(), (std::vector<std::string>{"conv1/7x7_s2_1"}));
}

OPENVINO_TEST(onnx_editor, subgraph__custom_input_name_already_exist) {
    const auto model_path = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                         SERIALIZED_ZOO,
                                                         "onnx/model_editor/subgraph__inception_head.onnx");

    ONNXModelEditor editor{model_path};
    try {
        editor.extract_subgraph({{InputEdge{1, 0, "conv1/7x7_s2_b_0"}}}, {});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("New custom input name: conv1/7x7_s2_b_0 already exist in the graph") !=
                    std::string::npos);
    }
}

// HIGHT LEVEL API TESTS
// INPUT EDGES TEST
OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_output_name_and_input_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    const InputEdge edge =
        editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_2"}}, EditorInput{"conv1/7x7_s2_1"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_1"}}, EditorInput{"data_0"});
    EXPECT_EQ(edge2.m_node_idx, 0);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_output_name_and_input_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    const InputEdge edge = editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_2"}}, EditorInput{0});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_1"}}, EditorInput{1});
    EXPECT_EQ(edge2.m_node_idx, 0);
    EXPECT_EQ(edge2.m_port_idx, 1);

    const InputEdge edge3 = editor.find_input_edge(EditorNode{EditorOutput{"conv1/7x7_s2_1"}}, EditorInput{2});
    EXPECT_EQ(edge3.m_node_idx, 0);
    EXPECT_EQ(edge3.m_port_idx, 2);
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_node_name_and_input_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    const InputEdge edge = editor.find_input_edge(EditorNode{"relu1"}, EditorInput{"conv1/7x7_s2_1"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"conv1"}, EditorInput{"conv1/7x7_s2_w_0"});
    EXPECT_EQ(edge2.m_node_idx, 0);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_node_name_and_input_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const InputEdge edge = editor.find_input_edge(EditorNode{"relu1_name"}, EditorInput{0});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"split_name"}, EditorInput{0});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_node_name_and_input_index_custom_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const InputEdge edge = editor.find_input_edge(EditorNode{"relu1_name"}, EditorInput{0, "custom_input_name_1"});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);
    EXPECT_EQ(edge.m_new_input_name, "custom_input_name_1");

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"split_name"}, EditorInput{0, "custom_input_name_2"});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 0);
    EXPECT_EQ(edge2.m_new_input_name, "custom_input_name_2");
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_node_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const InputEdge edge = editor.find_input_edge(EditorNode{0}, EditorInput{0, "custom_input_name_1"});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);
    EXPECT_EQ(edge.m_new_input_name, "custom_input_name_1");

    const InputEdge edge2 = editor.find_input_edge(EditorNode{5}, EditorInput{0});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 0);

    try {
        editor.find_input_edge(EditorNode{99}, EditorInput{"conv1/7x7_s2_1"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Provided node index: 99 is out of scope") != std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_empty_node_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    try {
        editor.find_input_edge(EditorNode{""}, EditorInput{"conv1/7x7_s2_1"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with name: not_given and output_name: not_given was not found") !=
                    std::string::npos);
    }
}

// OUTPUT EDGES TEST
OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_output_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{EditorOutput{"mul2"}}, EditorOutput{"mul2"});
    EXPECT_EQ(edge.m_node_idx, 4);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{EditorOutput{"split1"}}, EditorOutput{"split2"});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);

    // simplified overload
    const OutputEdge edge3 = editor.find_output_edge("mul2");
    EXPECT_EQ(edge3.m_node_idx, 4);
    EXPECT_EQ(edge3.m_port_idx, 0);

    const OutputEdge edge4 = editor.find_output_edge("split2");
    EXPECT_EQ(edge4.m_node_idx, 5);
    EXPECT_EQ(edge4.m_port_idx, 1);
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_output_name_and_output_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{EditorOutput{"add2"}}, EditorOutput{0});
    EXPECT_EQ(edge.m_node_idx, 3);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{EditorOutput{"split1"}}, EditorOutput{1});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);

    const OutputEdge edge3 = editor.find_output_edge(EditorNode{EditorOutput{"split2"}}, EditorOutput{0});
    EXPECT_EQ(edge3.m_node_idx, 5);
    EXPECT_EQ(edge3.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_node_name_and_output_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"relu1_name"}, EditorOutput{"relu1"});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"split_name"}, EditorOutput{"split2"});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_node_name_and_output_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"relu1_name"}, EditorOutput{0});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"split_name"}, EditorOutput{1});
    EXPECT_EQ(edge2.m_node_idx, 5);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_node_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{5}, EditorOutput{1});
    EXPECT_EQ(edge.m_node_idx, 5);
    EXPECT_EQ(edge.m_port_idx, 1);

    try {
        editor.find_output_edge(EditorNode{99}, EditorOutput{"conv1/7x7_s2_1"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Provided node index: 99 is out of scope") != std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_select_edge_const_network) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    const InputEdge edge = editor.find_input_edge(EditorNode{EditorOutput{"relu4"}}, EditorInput{0});
    EXPECT_EQ(edge.m_node_idx, 3);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"relu4_name"}, EditorOutput{0});
    EXPECT_EQ(edge2.m_node_idx, 3);
    EXPECT_EQ(edge2.m_port_idx, 0);

    const OutputEdge edge3 = editor.find_output_edge(EditorNode{"add1_name"}, EditorOutput{0});
    EXPECT_EQ(edge3.m_node_idx, 4);
    EXPECT_EQ(edge3.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_select_edge_error_handling) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    // node with given output name not found
    try {
        editor.find_input_edge(EditorNode{EditorOutput{"not_existed"}}, EditorInput{0});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with name: not_given and output_name: not_existed was not found") !=
                    std::string::npos);
    }

    // node with given name not found
    try {
        editor.find_input_edge(EditorNode{"not_existed"}, EditorInput{0});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with name: not_existed and output_name: not_given was not found") !=
                    std::string::npos);
    }

    // input index out of scope
    try {
        editor.find_input_edge(EditorNode{"relu4_name"}, EditorInput{1});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not input with index: 1") != std::string::npos);
    }

    // output index out of scope
    try {
        editor.find_output_edge(EditorNode{"relu4_name"}, EditorOutput{1});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not output with index: 1") != std::string::npos);
    }

    // input name not found
    try {
        editor.find_input_edge(EditorNode{"relu4_name"}, EditorInput{"not_existed"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not input with name: not_existed") != std::string::npos);
    }

    // output name not found
    try {
        editor.find_output_edge(EditorNode{"relu4_name"}, EditorOutput{"not_existed"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 3 has not output with name: not_existed") != std::string::npos);
    }
}

// Nodes with ambiguous node names tests
OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_ambiguous_node_name_but_matched_input) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    InputEdge edge = editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"in2"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 1);

    const InputEdge edge2 = editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"add1"});
    EXPECT_EQ(edge2.m_node_idx, 3);
    EXPECT_EQ(edge2.m_port_idx, 1);
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_ambiguous_node_name_and_not_matched_input) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    try {
        editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"in3"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Input edge described by: add_ambiguous_name and input name: in3 was not found") !=
                    std::string::npos);
    }

    try {
        editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{"relu1"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find(
                "Given node name: add_ambiguous_name and input name: relu1 are ambiguous to determine input edge") !=
            std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_select_input_edge_by_ambiguous_node_name_and_input_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    try {
        editor.find_input_edge(EditorNode{"add_ambiguous_name"}, EditorInput{0});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("Given node name: add_ambiguous_name and input index: 0 are ambiguous to determine input edge") !=
            std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_ambiguous_node_name_but_matched_output) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{"add1"});
    EXPECT_EQ(edge.m_node_idx, 1);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{"add2"});
    EXPECT_EQ(edge2.m_node_idx, 3);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_the_same_node_name_and_output_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_2.onnx")};

    const OutputEdge edge = editor.find_output_edge(EditorNode{"add1"}, EditorOutput{0});
    EXPECT_EQ(edge.m_node_idx, 0);
    EXPECT_EQ(edge.m_port_idx, 0);

    const OutputEdge edge2 = editor.find_output_edge(EditorNode{EditorOutput{"add1"}}, EditorOutput{0});
    EXPECT_EQ(edge2.m_node_idx, 4);
    EXPECT_EQ(edge2.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_ambiguous_node_name_and_not_matched_output) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    try {
        editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{"split2"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Output edge described by: add_ambiguous_name and output name: split2 was not found") !=
                    std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_select_output_edge_by_ambiguous_node_name_and_output_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    try {
        editor.find_output_edge(EditorNode{"add_ambiguous_name"}, EditorOutput{0});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find(
                "Given node name: add_ambiguous_name and output index: 0 are ambiguous to determine output edge") !=
            std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_use_edge_mapper_with_graph_cutter) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    // InputEdge{1, "in2"}
    const auto input_edge_1 = editor.find_input_edge(EditorNode(EditorOutput("add1")), EditorInput(1));
    // InputEdge{2, "in3"}
    const auto input_edge_2 = editor.find_input_edge(EditorNode(EditorOutput("conv1")), EditorInput(0));

    const auto output_edge = editor.find_output_edge(EditorNode(EditorOutput("mul2")), EditorOutput(0));
    // OutputEdge{4, "mul2"}
    editor.extract_subgraph({input_edge_1, input_edge_2}, {output_edge});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/"
                                                        "subgraph__existing_inputs_and_outputs_based_extraction.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;

    // check if mapper was updated after the model changed
    const auto input_edge_4 = editor.find_input_edge(EditorNode(EditorOutput("relu1")), EditorInput(0));
    EXPECT_EQ(input_edge_4.m_node_idx, 0);
    EXPECT_EQ(input_edge_4.m_port_idx, 0);

    const auto input_edge_5 = editor.find_input_edge(EditorNode(EditorOutput("add1")), EditorInput(1));
    EXPECT_EQ(input_edge_5.m_node_idx, 1);
    EXPECT_EQ(input_edge_5.m_port_idx, 1);

    const auto output_edge_3 = editor.find_output_edge("mul2");
    EXPECT_EQ(output_edge_3.m_node_idx, 3);
    EXPECT_EQ(output_edge_3.m_port_idx, 0);
}

OPENVINO_TEST(onnx_editor, editor_api_use_edge_mapper_with_graph_cutter_custom_names) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const auto input_edge_1 = editor.find_input_edge(EditorNode{EditorOutput{"mul2"}}, EditorInput{1, "new_name_1"});
    const auto input_edge_2 =
        editor.find_input_edge(EditorNode{EditorOutput{"split2"}}, EditorInput{"add2", "new_name_2"});

    editor.extract_subgraph({input_edge_1, input_edge_2}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__use_edge_mapper_with_graph_cutter_custom_names.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, editor_api_find_output_consumers) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

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

OPENVINO_TEST(onnx_editor, editor_api_find_output_consumers_empty_result) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const std::vector<InputEdge> output_consumers = editor.find_output_consumers("not_existed");
    EXPECT_EQ(output_consumers.size(), 0);
}

OPENVINO_TEST(onnx_editor, editor_api_inputs_with_the_same_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab.onnx")};

    std::vector<InputEdge> output_consumers = editor.find_output_consumers("X");
    EXPECT_EQ(output_consumers[0].m_node_idx, 1);
    EXPECT_EQ(output_consumers[0].m_port_idx, 0);
    EXPECT_EQ(output_consumers[1].m_node_idx, 1);
    EXPECT_EQ(output_consumers[1].m_port_idx, 1);
}

OPENVINO_TEST(onnx_editor, editor_api_find_output_consumers_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests_3.onnx")};
    const std::string output_name{"2891"};

    std::vector<InputEdge> output_consumers = editor.find_output_consumers(output_name);
    EXPECT_EQ(output_consumers[0].m_node_idx, 3);
    EXPECT_EQ(output_consumers[0].m_port_idx, 0);
    EXPECT_EQ(output_consumers[0].m_new_input_name, output_name);
    EXPECT_EQ(output_consumers[1].m_node_idx, 4);
    EXPECT_EQ(output_consumers[1].m_port_idx, 0);
    EXPECT_EQ(output_consumers[1].m_new_input_name, output_name);
}

OPENVINO_TEST(onnx_editor, editor_api_is_correct_and_unambiguous_node) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    bool is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"relu1"}});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"mul2"}});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"split2"}});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{"relu1_name"});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{2});
    EXPECT_EQ(is_correct_node, true);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{99});
    EXPECT_EQ(is_correct_node, false);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{EditorOutput{"in3"}});
    EXPECT_EQ(is_correct_node, false);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{"add_ambiguous_name"});
    EXPECT_EQ(is_correct_node, false);

    is_correct_node = editor.is_correct_and_unambiguous_node(EditorNode{"not_exist"});
    EXPECT_EQ(is_correct_node, false);
}

OPENVINO_TEST(onnx_editor, editor_api_get_node_index) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_EQ(editor.get_node_index(EditorNode{2}), 2);
    EXPECT_EQ(editor.get_node_index(EditorNode{EditorOutput{"relu1"}}), 0);
    EXPECT_EQ(editor.get_node_index(EditorNode{EditorOutput{"split2"}}), 5);
    EXPECT_EQ(editor.get_node_index(EditorNode{"relu1_name"}), 0);

    try {
        editor.get_node_index(EditorNode{99});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Provided node index: 99 is out of scope") != std::string::npos);
    }

    try {
        editor.get_node_index(EditorNode{"add_ambiguous_name"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find(
                "The node with name: add_ambiguous_name, output_name: not_given, node_index: not_given is ambiguous") !=
            std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, editor_api_input_edge_from_tensor_with_single_consumer) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab.onnx")};

    const auto edge = editor.find_input_edge(EditorNode{EditorOutput{"Y"}}, EditorInput{1});
    editor.extract_subgraph({edge}, {});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/"
                                     "subgraph__twice_input_edge_from_tensor_with_single_consumer.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, editor_api_input_edge_from_tensor_with_single_consumer_ambiguous) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab.onnx")};

    try {
        editor.find_input_edge(EditorNode{EditorOutput{"Y"}}, EditorInput{"X"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("Node with index: 1 has more than one inputs with name: X") != std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, values__append_one_initializer) {
    onnx_editor::ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/model_editor/add_1D.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", ngraph::op::Constant::create(element::i64, Shape{2}, {1, 2}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_input<int64_t>(Shape{2}, {5, 6});
    test_case.add_expected_output<int64_t>(Shape{2}, {6, 8});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__append_two_initializers_to_invalid) {
    onnx_editor::ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/model_editor/add_1D_invalid.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", ngraph::op::Constant::create(element::i64, Shape{2}, {4, 2}));
    in_vals.emplace("B", ngraph::op::Constant::create(element::i64, Shape{2}, {1, 3}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 5});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__modify_one_initializer) {
    onnx_editor::ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/add_1D_with_initializers.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("B", ngraph::op::Constant::create(element::i64, Shape{2}, {3, 4}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {4, 6});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__modify_two_initializers) {
    onnx_editor::ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/add_1D_with_initializers.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", ngraph::op::Constant::create(element::i64, Shape{2}, {3, 6}));
    in_vals.emplace("B", ngraph::op::Constant::create(element::i64, Shape{2}, {2, 1}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 7});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__no_inputs_modify_two_initializers) {
    onnx_editor::ONNXModelEditor editor{
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/add_1D_with_initializers_only.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", ngraph::op::Constant::create(element::i64, Shape{2}, {1, 2}));
    in_vals.emplace("B", ngraph::op::Constant::create(element::i64, Shape{2}, {11, 22}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_expected_output<int64_t>(Shape{2}, {12, 24});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__append_two_initializers_change_shape_type) {
    onnx_editor::ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/model_editor/add_1D.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("A", ngraph::op::Constant::create(element::i8, Shape{2, 1}, {-1, 1}));
    in_vals.emplace("B", ngraph::op::Constant::create(element::i8, Shape{2, 1}, {-2, 2}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_expected_output<int8_t>(Shape{2, 1}, {-3, 3});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__append_two_initializers_mixed_types) {
    onnx_editor::ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/gather_elements_float_3D_axis_2.onnx")};
    std::map<std::string, std::shared_ptr<ngraph::op::Constant>> in_vals;

    in_vals.emplace("data", ngraph::op::Constant::create(element::i16, Shape{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    in_vals.emplace("indices", ngraph::op::Constant::create(element::i32, Shape{2, 2, 1}, {0, 1, 0, 1}));
    editor.set_input_values(in_vals);

    const auto function = editor.get_function();
    auto test_case = ov::test::TestCase(function);
    test_case.add_expected_output<int16_t>(Shape{2, 2, 1}, {1, 4, 5, 8});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, read_model_from_stream) {
    std::string path = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                    SERIALIZED_ZOO,
                                                    "onnx/external_data/external_data.onnx");
    std::ifstream stream{path, std::ios::in | std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    ONNXModelEditor editor{stream, path};

    auto test_case = ov::test::TestCase(editor.get_function());
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();

    stream.close();
}

OPENVINO_TEST(onnx_editor, combined__cut_and_replace_shape) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph__inception_head.onnx")};

    const auto new_shape = PartialShape({1, 64, 112, 112});
    editor.extract_subgraph({{InputEdge(1, 0)}}, {});
    editor.set_input_shapes({{"conv1/7x7_s2_1", new_shape}});

    const auto ref_model =
        ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                     SERIALIZED_ZOO,
                                     "onnx/model_editor/reference/subgraph__linear_model_head_cut.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;

    const auto graph_inputs = editor.get_function()->get_parameters();
    EXPECT_TRUE(find_input(graph_inputs, "conv1/7x7_s2_1")->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, cut_operator_with_no_schema) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/unknown_input_value_info.onnx")};

    editor.extract_subgraph({{InputEdge{1, 0}}}, {});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/reference/unknown_input_value_info.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, get_source_tensor_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_EQ(editor.get_source_tensor_name(InputEdge{0, 0}), "in1");
    EXPECT_EQ(editor.get_source_tensor_name(InputEdge{1, 0}), "relu1");
    EXPECT_EQ(editor.get_source_tensor_name(InputEdge{1, 1}), "in2");
    const auto edge1 = editor.find_input_edge(EditorOutput{"conv1"}, 1);
    EXPECT_EQ(editor.get_source_tensor_name(edge1), "in4");
    const auto edge2 = editor.find_input_edge(EditorOutput{"split2"}, 0);
    EXPECT_EQ(editor.get_source_tensor_name(edge2), "add2");
    EXPECT_EQ(editor.get_source_tensor_name(InputEdge{999, 999}), "");
}

OPENVINO_TEST(onnx_editor, is_model_input) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_TRUE(editor.is_input(InputEdge{0, 0}));
    const auto edge1 = editor.find_input_edge(EditorOutput{"add1"}, 1);
    EXPECT_TRUE(editor.is_input(edge1));

    EXPECT_FALSE(editor.is_input(InputEdge{1, 2}));
    EXPECT_FALSE(editor.is_input(InputEdge{3, 0}));
    EXPECT_FALSE(editor.is_input(InputEdge{11, 0}));
    const auto edge2 = editor.find_input_edge(EditorOutput{"conv1"}, 2);
    EXPECT_FALSE(editor.is_input(edge2));
    EXPECT_FALSE(editor.is_input(InputEdge{2, 1}));  // initializer is not treated as input
    const auto edge3 = editor.find_input_edge(EditorOutput{"conv1"}, EditorInput{"in4"});
    EXPECT_FALSE(editor.is_input(edge3));
}

OPENVINO_TEST(onnx_editor, get_target_tensor_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_EQ(editor.get_target_tensor_name(OutputEdge{0, 0}), "relu1");
    EXPECT_EQ(editor.get_target_tensor_name(OutputEdge{1, 0}), "add1");
    EXPECT_EQ(editor.get_target_tensor_name(OutputEdge{4, 0}), "mul2");
    const auto edge1 = editor.find_output_edge("split1");
    EXPECT_EQ(editor.get_target_tensor_name(edge1), "split1");
    EXPECT_EQ(editor.get_target_tensor_name(OutputEdge{999, 999}), "");
}

OPENVINO_TEST(onnx_editor, is_model_output) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_TRUE(editor.is_output(OutputEdge{4, 0}));
    EXPECT_TRUE(editor.is_output(OutputEdge{5, 1}));
    const auto edge1 = editor.find_output_edge(EditorNode{"split_name"}, EditorOutput{"split2"});
    EXPECT_TRUE(editor.is_output(edge1));

    EXPECT_FALSE(editor.is_output(OutputEdge{4, 1}));
    EXPECT_FALSE(editor.is_output(OutputEdge{0, 0}));
    EXPECT_FALSE(editor.is_output(OutputEdge{11, 0}));
    const auto edge2 = editor.find_output_edge("add2");
    EXPECT_FALSE(editor.is_output(edge2));
}

OPENVINO_TEST(onnx_editor, model_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const auto inputs = editor.model_inputs();
    EXPECT_TRUE(inputs == (std::vector<std::string>{"in1", "in2", "in3"}));  // in4 is initializer
}

OPENVINO_TEST(onnx_editor, model_inputs_with_non_input_initializers) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/instance_norm_dynamic.onnx")};

    const auto inputs = editor.model_inputs();
    EXPECT_TRUE(inputs == (std::vector<std::string>{"input"}));
}

OPENVINO_TEST(onnx_editor, model_output) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const auto outputs = editor.model_outputs();
    EXPECT_TRUE(outputs == (std::vector<std::string>{"mul1", "split2", "mul2"}));
}

OPENVINO_TEST(onnx_editor, get_tensor_shape) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_EQ(editor.get_tensor_shape("mul2"), (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(editor.get_tensor_shape("in1"), (PartialShape{2, 2}));
    EXPECT_EQ(editor.get_tensor_shape("in2"), (PartialShape{}));
    EXPECT_EQ(editor.get_tensor_shape("in3"), (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(editor.get_tensor_shape("relu1"), (PartialShape{2, 2}));
    EXPECT_EQ(editor.get_tensor_shape("add1"), (PartialShape{2, 2}));
    try {
        editor.get_tensor_shape("not_existed");
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("The tensor: not_existed was not found in the graph") != std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, get_tensor_shape_after_modification) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_EQ(editor.get_tensor_shape("in3"), (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(editor.get_tensor_shape("conv1"), (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(editor.get_tensor_shape("mul2"), (PartialShape{1, 1, 2, 2}));
    editor.set_input_shapes({{"in3", (PartialShape{1, 1, 4, 4})}});
    EXPECT_EQ(editor.get_tensor_shape("conv1"), (PartialShape{1, 1, 4, 4}));
    EXPECT_EQ(editor.get_tensor_shape("in3"), (PartialShape{1, 1, 4, 4}));
}

OPENVINO_TEST(onnx_editor, is_correct_tensor_name) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_TRUE(editor.is_correct_tensor_name("in1"));
    EXPECT_TRUE(editor.is_correct_tensor_name("relu1"));
    EXPECT_TRUE(editor.is_correct_tensor_name("split2"));
    EXPECT_TRUE(editor.is_correct_tensor_name("mul2"));
    EXPECT_TRUE(editor.is_correct_tensor_name("in4"));

    EXPECT_FALSE(editor.is_correct_tensor_name("relu1_name"));
    EXPECT_FALSE(editor.is_correct_tensor_name("not_existed"));
    EXPECT_FALSE(editor.is_correct_tensor_name(""));
}

OPENVINO_TEST(onnx_editor, get_input_ports) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const auto ports_1 = editor.get_input_ports(EditorNode{"relu1_name"});
    EXPECT_EQ(ports_1.size(), 1);
    EXPECT_EQ(ports_1[0], "in1");
    const auto ports_2 = editor.get_input_ports(EditorNode{"split_name"});
    EXPECT_EQ(ports_2.size(), 1);
    EXPECT_EQ(ports_2[0], "add2");
    const auto ports_3 = editor.get_input_ports(EditorNode{EditorOutput{"add2"}});
    EXPECT_EQ(ports_3.size(), 2);
    EXPECT_EQ(ports_3[0], "relu1");
    EXPECT_EQ(ports_3[1], "add1");
    try {
        editor.get_input_ports(EditorNode{"add_ambiguous_name"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find(
                "The node with name: add_ambiguous_name, output_name: not_given, node_index: not_given is ambiguous") !=
            std::string::npos);
    }
    try {
        editor.get_input_ports(EditorNode{""});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("The node with name: not_given, output_name: not_given, node_index: not_given is ambiguous") !=
            std::string::npos);
    }
}
OPENVINO_TEST(onnx_editor, get_output_ports) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    const auto ports_1 = editor.get_output_ports(EditorNode{"relu1_name"});
    EXPECT_EQ(ports_1.size(), 1);
    EXPECT_EQ(ports_1[0], "relu1");
    const auto ports_2 = editor.get_output_ports(EditorNode{"split_name"});
    EXPECT_EQ(ports_2.size(), 2);
    EXPECT_EQ(ports_2[0], "split1");
    EXPECT_EQ(ports_2[1], "split2");
    const auto ports_3 = editor.get_output_ports(EditorNode{EditorOutput{"add2"}});
    EXPECT_EQ(ports_3.size(), 1);
    EXPECT_EQ(ports_3[0], "add2");
    try {
        editor.get_output_ports(EditorNode{"add_ambiguous_name"});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find(
                "The node with name: add_ambiguous_name, output_name: not_given, node_index: not_given is ambiguous") !=
            std::string::npos);
    }
    try {
        editor.get_output_ports(EditorNode{""});
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(
            msg.find("The node with name: not_given, output_name: not_given, node_index: not_given is ambiguous") !=
            std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, add_output) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_abc.onnx")};

    editor.add_output({OutputEdge{0, 0}});

    const auto edge1 = editor.find_output_edge(EditorNode{"add_node1"}, EditorOutput{"X"});
    EXPECT_TRUE(editor.is_output(edge1));
}

OPENVINO_TEST(onnx_editor, get_tensor_element_type) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    EXPECT_EQ(editor.get_input_type("in1"), (element::f32));
    EXPECT_EQ(editor.get_input_type("in2"), (element::f32));
    editor.set_input_types({{"in3", (element::f16)}});
    EXPECT_EQ(editor.get_input_type("in3"), (element::f16));
}

OPENVINO_TEST(onnx_editor, subgraph__cut_one_edge_and_merge_all_new_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    // InputEdge{1, "relu1"}
    const auto input_edge_1 = editor.find_input_edge(EditorNode(EditorOutput("add1")), EditorInput(0));

    editor.extract_subgraph({{input_edge_1}}, {}, true);

    const auto ref_model = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__cut_one_edge_and_merge_all_new_inputs.onnx");

    auto result = compare_onnx_models(editor.model_string(), ref_model);

    // InputEdge{5, "add2"}
    const auto input_edge_2 = editor.find_input_edge(EditorNode(EditorOutput("split1")), EditorInput(0));

    editor.extract_subgraph({{input_edge_2}}, {}, true);

    const auto ref_model1 = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__cut_one_another_edge_and_merge_all_new_inputs.onnx");

    result = compare_onnx_models(editor.model_string(), ref_model1);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__cut_two_edges_from_one_source_and_merge_all_new_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    // InputEdge{3, "relu1"}
    const auto input_edge_1 = editor.find_input_edge(EditorNode(EditorOutput("add2")), EditorInput(0));
    // InputEdge{6, "relu1"}
    const auto input_edge_2 = editor.find_input_edge(EditorNode(EditorOutput("mul1")), EditorInput(0));

    editor.extract_subgraph({{input_edge_1, input_edge_2}}, {}, true);

    const auto ref_model = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__cut_two_edges_from_one_source_and_merge_all_new_inputs.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__cut_two_edges_from_different_sources_and_merge_all_new_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    // InputEdge{3, "add1"}
    const auto input_edge_1 = editor.find_input_edge(EditorNode(EditorOutput("add2")), EditorInput(1));
    // InputEdge{6, "relu1"}
    const auto input_edge_2 = editor.find_input_edge(EditorNode(EditorOutput("mul1")), EditorInput(0));

    editor.extract_subgraph({{input_edge_1, input_edge_2}}, {}, true);

    const auto ref_model = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__cut_two_edges_from_different_sources_and_merge_all_new_inputs.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__cut_all_edges_from_one_source_and_merge_all_new_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    // InputEdge{3, "relu1"}
    const auto input_edge_1 = editor.find_input_edge(EditorNode(EditorOutput("add2")), EditorInput(0));
    // InputEdge{6, "relu1"}
    const auto input_edge_2 = editor.find_input_edge(EditorNode(EditorOutput("mul1")), EditorInput(0));
    // InputEdge{1, "relu1"}
    const auto input_edge_3 = editor.find_input_edge(EditorNode(EditorOutput("add1")), EditorInput(0));

    editor.extract_subgraph({{input_edge_1, input_edge_2, input_edge_3}}, {}, true);

    const auto ref_model = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__cut_all_edges_from_one_source_and_merge_all_new_inputs.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__cut_custom_edges_from_different_sources_and_merge_all_new_inputs) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/subgraph_extraction_tests.onnx")};

    // InputEdge{3, "relu1"}
    const auto input_edge_1 = editor.find_input_edge(EditorNode(EditorOutput("add2")), EditorInput(0));
    // InputEdge{4, "add1"}
    const auto input_edge_2 = editor.find_input_edge(EditorNode(EditorOutput("mul2")), EditorInput(0));
    // InputEdge{3, "add1"}
    const auto input_edge_3 = editor.find_input_edge(EditorNode(EditorOutput("add2")), EditorInput(1));

    editor.extract_subgraph({{input_edge_2, input_edge_1, input_edge_3}}, {}, true);

    const auto ref_model = ngraph::file_util::path_join(
        ov::test::utils::getExecutableDirectory(),
        SERIALIZED_ZOO,
        "onnx/model_editor/reference/subgraph__cut_custom_edges_from_different_sources_and_merge_all_new_inputs.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__duplicated_output) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab_duplicated_output.onnx")};

    const auto y_out_edge = editor.find_output_edge("Y");
    editor.extract_subgraph({}, {{y_out_edge}});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab.onnx");

    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, subgraph__duplicated_output_2) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab_duplicated_output.onnx")};

    const auto y_out_edge_1 = editor.find_output_edge("Y");
    const auto y_out_edge_2 = editor.find_output_edge("Y");
    editor.extract_subgraph({}, {{y_out_edge_1, y_out_edge_2}});

    const auto ref_model = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/add_ab_duplicated_output.onnx");

    // Model not changed
    const auto result = compare_onnx_models(editor.model_string(), ref_model);

    EXPECT_TRUE(result.is_ok) << result.error_message;
}

OPENVINO_TEST(onnx_editor, onnx_shape_infer_exception) {
    ONNXModelEditor editor{ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                        SERIALIZED_ZOO,
                                                        "onnx/model_editor/onnx_shape_infer_exception.onnx")};

    const auto input_edge = editor.find_input_edge(EditorNode(EditorOutput("input_ReduceMin")), EditorInput(0));

    EXPECT_NO_THROW(editor.extract_subgraph({{input_edge}}, {}));
}
