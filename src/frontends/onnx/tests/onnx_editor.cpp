// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_editor_test_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace ov::frontend;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

namespace {
using InputTypePred = std::function<bool(const std::shared_ptr<ov::Node>)>;

// A higher order factory function that produces predicates bound to a particular element type
InputTypePred element_type_is(const ov::element::Type et) {
    return [et](const std::shared_ptr<ov::Node> input) {
        return input->get_element_type() == et;
    };
}

std::shared_ptr<op::v0::Parameter> find_input(const ov::ParameterVector& inputs, const std::string& name) {
    const auto input_pos =
        std::find_if(std::begin(inputs), std::end(inputs), [&name](const ov::ParameterVector::value_type i) {
            return i->get_friendly_name() == name;
        });

    return *input_pos;
}
}  // namespace

OPENVINO_TEST(onnx_editor, types__single_input_type_substitution) {
    // the original model contains 2 inputs with i64 data type and one f32 input
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_abc.onnx", &front_end);

    input_model->set_element_type(input_model->get_place_by_tensor_name("A"), ov::element::i64);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    const auto float_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(ov::element::f32));

    const auto integer_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(ov::element::i64));

    EXPECT_EQ(float_inputs_count, 0);
    EXPECT_EQ(integer_inputs_count, 3);

    EXPECT_EQ(find_input(graph_inputs, "A")->get_element_type(), ov::element::i64);
}

OPENVINO_TEST(onnx_editor, types__all_inputs_type_substitution) {
    // the original model contains 2 inputs with i64 data type and one f32 input
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_abc.onnx", &front_end);

    input_model->set_element_type(input_model->get_place_by_tensor_name("A"), ov::element::i8);
    input_model->set_element_type(input_model->get_place_by_tensor_name("B"), ov::element::i8);
    input_model->set_element_type(input_model->get_place_by_tensor_name("C"), ov::element::i8);

    const auto model = front_end->convert(input_model);

    const auto graph_inputs = model->get_parameters();

    const auto float_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(ov::element::f32));

    const auto integer_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(ov::element::i8));

    EXPECT_EQ(float_inputs_count, 0);
    EXPECT_EQ(integer_inputs_count, 3);
}

OPENVINO_TEST(onnx_editor, types__missing_type_in_input_descriptor) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/invalid_input_no_type.onnx");

    // input A doesn't have the "type" field in the model and so the data type cannot be modified
    EXPECT_THROW(input_model->set_element_type(input_model->get_place_by_tensor_name("A"), ov::element::f32),
                 ov::Exception);
}

OPENVINO_TEST(onnx_editor, types__missing_tensor_type_in_input_descriptor) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/invalid_input_no_tensor_type.onnx");

    // input A doesn't have the "tensor_type" field in the model
    EXPECT_THROW(input_model->set_element_type(input_model->get_place_by_tensor_name("A"), ov::element::f32),
                 ov::Exception);
}

OPENVINO_TEST(onnx_editor, types__unsupported_data_type_passed) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/add_abc.onnx");

    EXPECT_THROW(input_model->set_element_type(input_model->get_place_by_tensor_name("A"), ov::element::dynamic),
                 ov::Exception);
}

OPENVINO_TEST(onnx_editor, types__incorrect_input_name_passed) {
    auto input_model = load_model("model_editor/add_abc.onnx");

    EXPECT_EQ(input_model->get_place_by_tensor_name("ShiaLaBeouf"), nullptr);
}

OPENVINO_TEST(onnx_editor, types__elem_type_missing_in_input) {
    // the original model contains 2 inputs with i64 data type and one f32 input
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/elem_type_missing_in_input.onnx", &front_end);

    // the "elem_type" is missing in the model but it should be possible to set the type anyway
    EXPECT_NO_THROW(input_model->set_element_type(input_model->get_place_by_tensor_name("A"), ov::element::i64));

    const auto model = front_end->convert(input_model);

    const auto graph_inputs = model->get_parameters();

    const auto integer_inputs_count =
        std::count_if(std::begin(graph_inputs), std::end(graph_inputs), element_type_is(ov::element::i64));

    EXPECT_EQ(integer_inputs_count, 2);

    const auto function_result = model->get_result();
    EXPECT_EQ(function_result->get_element_type(), ov::element::i64);
}

OPENVINO_TEST(onnx_editor, shapes__modify_single_input) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__add_two_inputs.onnx", &front_end);

    const auto new_shape = PartialShape{1};

    input_model->set_partial_shape(input_model->get_place_by_tensor_name("B"), new_shape);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "B")->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, shapes__modify_all_inputs) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__add_two_inputs.onnx", &front_end);

    const auto new_shape = PartialShape{1, 2, 3, 5, 8, 13};

    input_model->set_partial_shape(input_model->get_place_by_tensor_name("A"), new_shape);
    input_model->set_partial_shape(input_model->get_place_by_tensor_name("B"), new_shape);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    for (const auto& input : graph_inputs) {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

OPENVINO_TEST(onnx_editor, shapes__dynamic_rank_in_model) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__dynamic_rank_in_model.onnx", &front_end);

    // input A in the model doesn't have the "shape" field meaning it has dynamic rank
    // it should still be possible to set such input's shape to some custom value
    const auto expected_shape_of_A = PartialShape{1, 2};
    EXPECT_NO_THROW(input_model->set_partial_shape(input_model->get_place_by_tensor_name("A"), expected_shape_of_A));

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "A")->get_partial_shape().same_scheme(expected_shape_of_A));
}

OPENVINO_TEST(onnx_editor, shapes__set_dynamic_dimension) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__add_two_inputs.onnx", &front_end);

    const auto new_shape = PartialShape{Dimension::dynamic()};

    input_model->set_partial_shape(input_model->get_place_by_tensor_name("A"), new_shape);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    EXPECT_TRUE(find_input(graph_inputs, "A")->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, shapes__set_mixed_dimensions) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__add_two_inputs.onnx", &front_end);

    const auto new_shape_A = PartialShape{21, Dimension::dynamic()};
    const auto new_shape_B = PartialShape{Dimension::dynamic(), 37};

    input_model->set_partial_shape(input_model->get_place_by_tensor_name("A"), new_shape_A);
    input_model->set_partial_shape(input_model->get_place_by_tensor_name("B"), new_shape_B);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape_A));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape_B));
}

OPENVINO_TEST(onnx_editor, shapes__set_scalar_inputs) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__add_two_inputs.onnx", &front_end);

    const auto new_shape = PartialShape{};

    input_model->set_partial_shape(input_model->get_place_by_tensor_name("A"), new_shape);
    input_model->set_partial_shape(input_model->get_place_by_tensor_name("B"), new_shape);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    const auto input_A = find_input(graph_inputs, "A");
    EXPECT_TRUE(input_A->get_partial_shape().same_scheme(new_shape));

    const auto input_B = find_input(graph_inputs, "B");
    EXPECT_TRUE(input_B->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, shapes__static_to_dynamic_rank_substitution) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/shapes__add_two_inputs.onnx", &front_end);

    const auto new_shape = PartialShape::dynamic();

    input_model->set_partial_shape(input_model->get_place_by_tensor_name("A"), new_shape);
    input_model->set_partial_shape(input_model->get_place_by_tensor_name("B"), new_shape);

    const auto model = front_end->convert(input_model);
    const auto graph_inputs = model->get_parameters();

    for (const auto& input : graph_inputs) {
        EXPECT_TRUE(input->get_partial_shape().same_scheme(new_shape));
    }
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_head_cut) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_operation_name("relu1")}, {});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_head_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_head_cut_ins_and_outs) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_operation_name("relu1")}, {input_model->get_outputs()[0]});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_head_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_deeper_head_cut) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_operation_name("maxpool1")}, {});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_deeper_head_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_tail_cut) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({}, {input_model->get_place_by_operation_name("relu1")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_tail_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_tail_cut_ins_and_outs) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_inputs()[0]}, {input_model->get_place_by_operation_name("relu1")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_tail_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_with_initializer_tail_cut) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head_with_initializer.onnx", &front_end);

    input_model->extract_subgraph({}, {input_model->get_place_by_tensor_name("conv1/7x7_s2_2")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_with_initializer_tail_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__initializer_without_matching_input_tail_cut) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__initializer_without_matching_input.onnx", &front_end);

    input_model->extract_subgraph({}, {input_model->get_place_by_tensor_name("conv1/7x7_s2_2")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__initializer_without_matching_input_tail_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__linear_model_deeper_tail_cut) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({}, {input_model->get_place_by_tensor_name("conv1/7x7_s2_1")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_deeper_tail_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__no_input_params) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    input_model->extract_subgraph({}, {});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/subgraph__inception_head.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__initializer_to_input_replacement) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head_with_initializer.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_tensor_name("conv1/7x7_s2_b_0")},
                                  {input_model->get_place_by_tensor_name("conv1/7x7_s2_1")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__initializer_to_input_replacement.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__initializer_to_input_replacement_2) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__initializer_without_matching_input.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_tensor_name("conv1/7x7_s2_b_0")},
                                  {input_model->get_place_by_tensor_name("conv1/7x7_s2_1")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__initializer_to_input_replacement.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiout_op_output_edge) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    input_model->extract_subgraph({}, {input_model->get_place_by_tensor_name("split2")});

    auto model = front_end->convert(input_model);
    auto model_ref = convert_model("model_editor/reference/subgraph__multiout_op_output_edge.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__existing_inputs_and_outputs_based_extraction) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    input_model->extract_subgraph(
        {input_model->get_place_by_tensor_name("in1"), input_model->get_place_by_tensor_name("in3")},
        {input_model->get_place_by_tensor_name("mul2")});

    auto model = front_end->convert(input_model);
    auto model_ref =
        convert_model("model_editor/reference/subgraph__existing_inputs_and_outputs_based_extraction.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__twice_input_edge_from_tensor_with_single_consumer) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_ab.onnx", &front_end);

    auto x_place = input_model->get_place_by_tensor_name("X");
    ASSERT_TRUE(x_place != nullptr) << "get_place_by_tensor_name(\"X\") returned nullptr";
    auto x_consumers = x_place->get_consuming_operations();
    ASSERT_FALSE(x_consumers.empty()) << "get_consuming_operations() returned empty container for \"X\"";

    input_model->extract_subgraph({x_consumers[0]->get_input_port(1)}, {});

    auto model = front_end->convert(input_model);
    auto model_ref =
        convert_model("model_editor/reference/subgraph__twice_input_edge_from_tensor_with_single_consumer.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    auto relu_node = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(relu_node != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    auto relu_consumers = relu_node->get_consuming_operations();

    ASSERT_GE(relu_consumers.size(), 3) << "relu_consumers vector does not have enough elements";
    ASSERT_TRUE(relu_consumers[0] != nullptr) << "relu_consumers[0] is nullptr";
    ASSERT_TRUE(relu_consumers[2] != nullptr) << "relu_consumers[2] is nullptr";

    input_model->extract_subgraph(
        {relu_consumers[0]->get_input_port(0), relu_consumers[2]->get_input_port(0)},
        {input_model->get_place_by_tensor_name("mul1"), input_model->get_place_by_tensor_name("mul2")});
    auto model = front_end->convert(input_model);

    auto model_ref =
        convert_model("model_editor/reference/subgraph__input_edge_from_tensor_with_multiple_consumers.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_2) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    auto relu_node = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(relu_node != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    auto relu_consumers = relu_node->get_consuming_operations();
    ASSERT_GE(relu_consumers.size(), 2) << "relu_consumers vector does not have enough elements";
    ASSERT_TRUE(relu_consumers[1] != nullptr) << "relu_consumers[1] is nullptr";

    input_model->extract_subgraph(
        {relu_consumers[1]->get_input_port(0), relu_consumers[1]->get_input_port(1)},
        {input_model->get_place_by_tensor_name("mul2"), relu_consumers[1]->get_output_port()});
    auto model = front_end->convert(input_model);

    auto model_ref =
        convert_model("model_editor/reference/subgraph__input_edge_from_tensor_with_multiple_consumers_2.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_3) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    auto relu_node = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(relu_node != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    auto relu_consumers = relu_node->get_consuming_operations();
    ASSERT_GE(relu_consumers.size(), 3) << "relu_consumers vector does not have enough elements";
    ASSERT_TRUE(relu_consumers[1] != nullptr) << "relu_consumers[1] is nullptr";
    ASSERT_TRUE(relu_consumers[2] != nullptr) << "relu_consumers[2] is nullptr";

    input_model->extract_subgraph(
        {relu_consumers[1]->get_input_port(0), relu_consumers[2]->get_input_port(0)},
        {input_model->get_place_by_tensor_name("mul1"), input_model->get_place_by_tensor_name("split2")});
    auto model = front_end->convert(input_model);

    auto model_ref =
        convert_model("model_editor/reference/subgraph__input_edge_from_tensor_with_multiple_consumers_3.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_4) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    auto relu_node = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(relu_node != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    auto relu_consumers = relu_node->get_consuming_operations();
    ASSERT_GE(relu_consumers.size(), 2) << "relu_consumers vector does not have enough elements";
    ASSERT_TRUE(relu_consumers[0] != nullptr) << "relu_consumers[0] is nullptr";
    ASSERT_TRUE(relu_consumers[1] != nullptr) << "relu_consumers[1] is nullptr";

    input_model->extract_subgraph({relu_consumers[0]->get_input_port(0), relu_consumers[1]->get_input_port(0)}, {});
    auto model = front_end->convert(input_model);

    auto model_ref =
        convert_model("model_editor/reference/subgraph__input_edge_from_tensor_with_multiple_consumers_4.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_5) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    auto relu_node = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(relu_node != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    auto relu_consumers = relu_node->get_consuming_operations();
    ASSERT_GE(relu_consumers.size(), 2) << "relu_consumers vector does not have enough elements";
    ASSERT_TRUE(relu_consumers[1] != nullptr) << "relu_consumers[1] is nullptr";

    input_model->extract_subgraph(
        {relu_consumers[1]->get_input_port(0)},
        {input_model->get_place_by_tensor_name("mul1"), input_model->get_place_by_tensor_name("split2")});
    auto model = front_end->convert(input_model);

    auto model_ref =
        convert_model("model_editor/reference/subgraph__input_edge_from_tensor_with_multiple_consumers_5.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__input_edge_from_tensor_with_multiple_consumers_custom_names) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);

    auto relu_node = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(relu_node != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    auto relu_consumers = relu_node->get_consuming_operations();

    ASSERT_GE(relu_consumers.size(), 3) << "relu_consumers vector does not have enough elements";
    ASSERT_TRUE(relu_consumers[0] != nullptr) << "relu_consumers[0] is nullptr";
    ASSERT_TRUE(relu_consumers[2] != nullptr) << "relu_consumers[2] is nullptr";
    ASSERT_TRUE(relu_consumers[0]->get_input_port(0) != nullptr) << "relu_consumers[0]->get_input_port(0) is nullptr";
    ASSERT_TRUE(relu_consumers[2]->get_input_port(0) != nullptr) << "relu_consumers[2]->get_input_port(0) is nullptr";

    input_model->cut_and_add_new_input(relu_consumers[0]->get_input_port(0), "new_name_1");
    input_model->cut_and_add_new_input(relu_consumers[2]->get_input_port(0), "new_name_2");

    input_model->extract_subgraph(
        {},
        {input_model->get_place_by_tensor_name("mul1"), input_model->get_place_by_tensor_name("mul2")});

    auto model = front_end->convert(input_model);

    auto model_ref = convert_model(
        "model_editor/reference/subgraph__input_edge_from_tensor_with_multiple_consumers_custom_names.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_input_relu2) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests_2.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_tensor_name("relu1")}, {});

    auto model = front_end->convert(input_model);

    auto model_ref = convert_model("model_editor/reference/subgraph__multiple_consumers_of_graph_input_relu2.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests_2.onnx", &front_end);

    input_model->extract_subgraph({input_model->get_place_by_tensor_name("in2")}, {});

    auto model = front_end->convert(input_model);

    auto model_ref = convert_model("model_editor/reference/subgraph__multiple_consumers_of_graph_initializer.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer_2) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests_2.onnx", &front_end);

    input_model->extract_subgraph(
        {input_model->get_place_by_tensor_name("in2"), input_model->get_place_by_tensor_name("in1")},
        {});

    auto model = front_end->convert(input_model);

    auto model_ref = convert_model("model_editor/reference/subgraph__multiple_consumers_of_graph_initializer.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__multiple_consumers_of_graph_initializer_relu2_and_init) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests_2.onnx", &front_end);

    auto in2_place = input_model->get_place_by_tensor_name("in2");
    ASSERT_TRUE(in2_place != nullptr) << "get_place_by_tensor_name(\"in2\") returned nullptr";
    auto relu3_place = input_model->get_place_by_tensor_name("relu3");
    ASSERT_TRUE(relu3_place != nullptr) << "get_place_by_tensor_name(\"relu3\") returned nullptr";
    auto relu3_consumers = relu3_place->get_consuming_operations();
    ASSERT_FALSE(relu3_consumers.empty()) << "get_consuming_operations() returned empty container for \"relu3\"";
    auto relu3_input_port = relu3_consumers[0]->get_input_port(0);
    ASSERT_TRUE(relu3_input_port != nullptr) << "get_input_port(0) returned nullptr for relu3's first consumer";

    input_model->extract_subgraph({in2_place, relu3_input_port}, {});

    auto model = front_end->convert(input_model);

    auto model_ref =
        convert_model("model_editor/reference/subgraph__multiple_consumers_of_graph_initializer_relu2_and_init.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__inputs_getter) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    auto inputs = input_model->get_inputs();
    auto inputs_ref = std::vector<std::string>{"data_0", "conv1/7x7_s2_w_0", "conv1/7x7_s2_b_0"};

    EXPECT_EQ(inputs.size(), inputs_ref.size());
    for (size_t idx = 0; idx < inputs_ref.size(); ++idx) {
        EXPECT_EQ(inputs[idx]->get_names()[0], inputs_ref[idx]);
    }

    input_model->extract_subgraph({input_model->get_place_by_tensor_name("conv1/7x7_s2_1")}, {});

    inputs = input_model->get_inputs();
    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(inputs[0]->get_names()[0], "conv1/7x7_s2_1");
}

OPENVINO_TEST(onnx_editor, subgraph__custom_input_name_already_exist) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);
    try {
        input_model->cut_and_add_new_input(input_model->get_place_by_operation_name("relu1"), "conv1/7x7_s2_b_0");
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("The name 'conv1/7x7_s2_b_0' is already used by another tensor.") != std::string::npos);
    }
}

OPENVINO_TEST(onnx_editor, values__append_one_initializer) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_1D.onnx", &front_end);

    auto place = input_model->get_place_by_tensor_name("A");
    input_model->set_tensor_value(place, std::vector<int64_t>{1, 2}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_input<int64_t>(Shape{2}, {5, 6});
    test_case.add_expected_output<int64_t>(Shape{2}, {6, 8});
    test_case.run();
}

/*
// Not applicable for InputModel
OPENVINO_TEST(onnx_editor, values__append_two_initializers_to_invalid) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_1D_invalid.onnx", &front_end);
    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> in_vals;
    // in_vals.emplace("A", ov::op::v0::Constant::create( ov::element::i64, Shape{2}, {4, 2}));
    // in_vals.emplace("B", ov::op::v0::Constant::create( ov::element::i64, Shape{2}, {1, 3}));
    // editor.set_input_values(in_vals);

    auto place = input_model->get_place_by_operation_name_and_input_port("add_node", 0);
    input_model->set_tensor_value(place, std::vector<int64_t>{3, 2}.data());

    place = input_model->get_place_by_operation_name_and_input_port("add_node", 1);
    input_model->set_tensor_value(place, std::vector<int64_t>{1, 3}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 5});
    test_case.run();
}
*/

OPENVINO_TEST(onnx_editor, values__modify_one_initializer) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_1D_with_initializers.onnx", &front_end);

    auto place = input_model->get_place_by_tensor_name("B");
    input_model->set_tensor_value(place, std::vector<int64_t>{3, 4}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_expected_output<int64_t>(Shape{2}, {4, 6});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__modify_two_initializers) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_1D_with_initializers.onnx", &front_end);

    auto place = input_model->get_place_by_tensor_name("A");
    input_model->set_tensor_value(place, std::vector<int64_t>{3, 6}.data());

    place = input_model->get_place_by_tensor_name("B");
    input_model->set_tensor_value(place, std::vector<int64_t>{2, 1}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 7});
    test_case.run();
}

/*
// Not applicable for InputModel
OPENVINO_TEST(onnx_editor, values__no_inputs_modify_two_initializers) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_1D_with_initializers_only.onnx", &front_end);
    std::map<std::string, std::shared_ptr<ov::op::v0::Constant>> in_vals;

    auto place = input_model->get_place_by_tensor_name("A");
    input_model->set_tensor_value(place, std::vector<int64_t>{1, 2}.data());

    place = input_model->get_place_by_tensor_name("B");
    input_model->set_tensor_value(place, std::vector<int64_t>{11, 22}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_expected_output<int64_t>(Shape{2}, {12, 24});
    test_case.run();
}
*/

OPENVINO_TEST(onnx_editor, values__append_two_initializers_change_shape_type) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_1D.onnx", &front_end);

    auto place = input_model->get_place_by_tensor_name("A");
    input_model->set_element_type(place, ov::element::i8);
    input_model->set_partial_shape(place, Shape{2, 1});
    input_model->set_tensor_value(place, std::vector<int8_t>{-1, 1}.data());

    place = input_model->get_place_by_tensor_name("B");
    input_model->set_element_type(place, ov::element::i8);
    input_model->set_partial_shape(place, Shape{2, 1});
    input_model->set_tensor_value(place, std::vector<int8_t>{-2, 2}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_expected_output<int8_t>(Shape{2, 1}, {-3, 3});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, values__append_two_initializers_mixed_types) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("gather_elements_float_3D_axis_2.onnx", &front_end);
    auto place = input_model->get_place_by_tensor_name("data");
    input_model->set_element_type(place, ov::element::i16);
    input_model->set_partial_shape(place, Shape{2, 2, 2});
    input_model->set_tensor_value(place, std::vector<int16_t>{1, 2, 3, 4, 5, 6, 7, 8}.data());

    place = input_model->get_place_by_tensor_name("indices");
    input_model->set_element_type(place, ov::element::i32);
    input_model->set_partial_shape(place, Shape{2, 2, 1});
    input_model->set_tensor_value(place, std::vector<int32_t>{0, 1, 0, 1}.data());

    const auto model = front_end->convert(input_model);
    auto test_case = ov::test::TestCase(model);
    test_case.add_expected_output<int16_t>(Shape{2, 2, 1}, {1, 4, 5, 8});
    test_case.run();
}

OPENVINO_TEST(onnx_editor, combined__cut_and_replace_shape) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph__inception_head.onnx", &front_end);

    const auto new_shape = PartialShape({1, 64, 112, 112});
    auto place = input_model->get_place_by_tensor_name("conv1/7x7_s2_1");
    input_model->extract_subgraph({place}, {});
    input_model->set_partial_shape(place, new_shape);

    auto model = front_end->convert(input_model);
    const auto model_ref = convert_model("model_editor/reference/subgraph__linear_model_head_cut.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;

    const auto graph_inputs = model->get_parameters();
    EXPECT_TRUE(find_input(graph_inputs, "conv1/7x7_s2_1")->get_partial_shape().same_scheme(new_shape));
}

OPENVINO_TEST(onnx_editor, cut_operator_with_no_schema) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/unknown_input_value_info.onnx", &front_end);
    input_model->extract_subgraph({input_model->get_place_by_tensor_name("X")}, {});

    auto model = front_end->convert(input_model);
    const auto model_ref = convert_model("model_editor/reference/unknown_input_value_info.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, is_model_input) {
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");

    auto in2_place = input_model->get_place_by_tensor_name("in2");
    ASSERT_TRUE(in2_place != nullptr) << "get_place_by_tensor_name(\"in2\") returned nullptr";
    EXPECT_TRUE(in2_place->is_input());

    auto conv1_place = input_model->get_place_by_tensor_name("conv1");
    ASSERT_TRUE(conv1_place != nullptr) << "get_place_by_tensor_name(\"conv1\") returned nullptr";
    EXPECT_FALSE(conv1_place->is_input());
}

OPENVINO_TEST(onnx_editor, is_model_output) {
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");

    auto split2_place = input_model->get_place_by_tensor_name("split2");
    ASSERT_TRUE(split2_place != nullptr) << "get_place_by_tensor_name(\"split2\") returned nullptr";
    EXPECT_TRUE(split2_place->is_output());

    auto add2_place = input_model->get_place_by_tensor_name("add2");
    ASSERT_TRUE(add2_place != nullptr) << "get_place_by_tensor_name(\"add2\") returned nullptr";
    EXPECT_FALSE(add2_place->is_output());
}

OPENVINO_TEST(onnx_editor, model_inputs) {
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");

    auto inputs = input_model->get_inputs();
    auto inputs_ref = std::vector<std::string>{"in1", "in2", "in3"};

    EXPECT_EQ(inputs.size(), inputs_ref.size());
    for (size_t idx = 0; idx < inputs_ref.size(); ++idx) {
        EXPECT_EQ(inputs[idx]->get_names()[0], inputs_ref[idx]);
    }
}

OPENVINO_TEST(onnx_editor, model_inputs_with_non_input_initializers) {
    auto input_model = load_model("instance_norm_dynamic.onnx");
    auto inputs = input_model->get_inputs();
    auto inputs_ref = std::vector<std::string>{"input"};

    EXPECT_EQ(inputs.size(), inputs_ref.size());
    for (size_t idx = 0; idx < inputs_ref.size(); ++idx) {
        EXPECT_EQ(inputs[idx]->get_names()[0], inputs_ref[idx]);
    }
}

OPENVINO_TEST(onnx_editor, model_output) {
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");
    auto outputs = input_model->get_outputs();
    auto outputs_ref = std::vector<std::string>{"mul1", "split2", "mul2"};

    EXPECT_EQ(outputs.size(), outputs_ref.size());
    for (size_t idx = 0; idx < outputs_ref.size(); ++idx) {
        EXPECT_EQ(outputs[idx]->get_names()[0], outputs_ref[idx]);
    }
}

OPENVINO_TEST(onnx_editor, get_tensor_shape) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("mul2")),
              (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("in1")), (PartialShape{2, 2}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("in2")), (PartialShape{}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("in3")), (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("relu1")), (PartialShape{2, 2}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("add1")), (PartialShape{2, 2}));
    try {
        input_model->get_partial_shape(input_model->get_place_by_tensor_name("not_existed"));
    } catch (const std::exception& e) {
        std::string msg{e.what()};
        EXPECT_TRUE(msg.find("expects a pointer") != std::string::npos);
    }
    EXPECT_THROW(input_model->get_partial_shape(nullptr), ov::Exception);
}

OPENVINO_TEST(onnx_editor, get_tensor_shape_after_modification) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("in3")), (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("conv1")),
              (PartialShape{1, 1, 2, 2}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("mul2")),
              (PartialShape{1, 1, 2, 2}));
    input_model->set_partial_shape(input_model->get_place_by_tensor_name("in3"), PartialShape{1, 1, 4, 4});
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("conv1")),
              (PartialShape{1, 1, 4, 4}));
    EXPECT_EQ(input_model->get_partial_shape(input_model->get_place_by_tensor_name("in3")), (PartialShape{1, 1, 4, 4}));
}

OPENVINO_TEST(onnx_editor, is_correct_tensor_name) {
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx", &front_end);
    EXPECT_TRUE(input_model->get_place_by_tensor_name("in1"));
    EXPECT_TRUE(input_model->get_place_by_tensor_name("relu1"));
    EXPECT_TRUE(input_model->get_place_by_tensor_name("split2"));
    EXPECT_TRUE(input_model->get_place_by_tensor_name("mul2"));
    EXPECT_TRUE(input_model->get_place_by_tensor_name("in4"));
    EXPECT_FALSE(input_model->get_place_by_operation_name("add_ambiguous_name"));
    EXPECT_FALSE(input_model->get_place_by_operation_name(""));

    EXPECT_FALSE(input_model->get_place_by_tensor_name("relu1_name"));
    EXPECT_FALSE(input_model->get_place_by_tensor_name("not_existed"));
    EXPECT_FALSE(input_model->get_place_by_tensor_name(""));
}

OPENVINO_TEST(onnx_editor, get_input_ports) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");
    const auto ports_1 = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(ports_1 != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    EXPECT_EQ(ports_1->get_input_port()->get_source_tensor()->get_names()[0], "in1");
    EXPECT_FALSE(ports_1->get_input_port(1));
    const auto ports_2 = input_model->get_place_by_operation_name("split_name");
    ASSERT_TRUE(ports_2 != nullptr) << "get_place_by_operation_name(\"split_name\") returned nullptr";
    EXPECT_EQ(ports_2->get_input_port(0)->get_source_tensor()->get_names()[0], "add2");
    EXPECT_FALSE(ports_2->get_input_port(1));
    const auto add2_tensor = input_model->get_place_by_tensor_name("add2");
    ASSERT_TRUE(add2_tensor != nullptr) << "get_place_by_tensor_name(\"add2\") returned nullptr";
    const auto ports_3 = add2_tensor->get_producing_operation();
    ASSERT_TRUE(ports_3 != nullptr) << "get_producing_operation() for \"add2\" returned nullptr";
    EXPECT_EQ(ports_3->get_input_port(0)->get_source_tensor()->get_names()[0], "relu1");
    EXPECT_EQ(ports_3->get_input_port(1)->get_source_tensor()->get_names()[0], "add1");
    EXPECT_FALSE(ports_3->get_input_port(2));
}

OPENVINO_TEST(onnx_editor, get_output_ports) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");
    const auto ports_1 = input_model->get_place_by_operation_name("relu1_name");
    ASSERT_TRUE(ports_1 != nullptr) << "get_place_by_operation_name(\"relu1_name\") returned nullptr";
    const auto relu1_out_port_0 = ports_1->get_output_port(0);
    ASSERT_TRUE(relu1_out_port_0 != nullptr) << "get_output_port(0) for relu1_name returned nullptr";
    const auto relu1_target_tensor = relu1_out_port_0->get_target_tensor();
    ASSERT_TRUE(relu1_target_tensor != nullptr) << "get_target_tensor() for relu1_name output port 0 returned nullptr";
    EXPECT_EQ(relu1_target_tensor->get_names()[0], "relu1");
    EXPECT_FALSE(ports_1->get_output_port(1));

    const auto ports_2 = input_model->get_place_by_operation_name("split_name");
    EXPECT_TRUE(ports_2 != nullptr) << "get_place_by_operation_name(\"split_name\") returned nullptr";
    const auto split_out_port_0 = ports_2->get_output_port(0);
    ASSERT_TRUE(split_out_port_0 != nullptr) << "get_output_port(0) for split_name returned nullptr";
    const auto split_target_tensor_0 = split_out_port_0->get_target_tensor();
    ASSERT_TRUE(split_target_tensor_0 != nullptr)
        << "get_target_tensor() for split_name output port 0 returned nullptr";
    EXPECT_EQ(split_target_tensor_0->get_names()[0], "split1");
    const auto split_out_port_1 = ports_2->get_output_port(1);
    ASSERT_TRUE(split_out_port_1 != nullptr) << "get_output_port(1) for split_name returned nullptr";
    const auto split_target_tensor_1 = split_out_port_1->get_target_tensor();
    ASSERT_TRUE(split_target_tensor_1 != nullptr)
        << "get_target_tensor() for split_name output port 1 returned nullptr";
    EXPECT_EQ(split_target_tensor_1->get_names()[0], "split2");
    EXPECT_FALSE(ports_2->get_output_port(2));

    const auto add2_tensor = input_model->get_place_by_tensor_name("add2");
    EXPECT_TRUE(add2_tensor != nullptr) << "get_place_by_tensor_name(\"add2\") returned nullptr";
    const auto ports_3 = add2_tensor->get_producing_operation();
    EXPECT_TRUE(ports_3 != nullptr) << "get_producing_operation() for \"add2\" returned nullptr";
    const auto add2_out_port_0 = ports_3->get_output_port();
    ASSERT_TRUE(add2_out_port_0 != nullptr) << "get_output_port() for add2 producing operation returned nullptr";
    const auto add2_target_tensor = add2_out_port_0->get_target_tensor();
    ASSERT_TRUE(add2_target_tensor != nullptr) << "get_target_tensor() for add2 output port returned nullptr";
    EXPECT_EQ(add2_target_tensor->get_names()[0], "add2");
    EXPECT_FALSE(ports_3->get_output_port(1));
}

OPENVINO_TEST(onnx_editor, add_output) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/add_abc.onnx");

    auto add_node1_place = input_model->get_place_by_operation_name("add_node1");
    ASSERT_TRUE(add_node1_place != nullptr) << "get_place_by_operation_name(\"add_node1\") returned nullptr";
    input_model->add_output(add_node1_place->get_target_tensor());

    EXPECT_EQ(input_model->get_outputs().size(), 2);

    EXPECT_THROW(input_model->add_output(nullptr), ov::Exception);
}

OPENVINO_TEST(onnx_editor, get_tensor_element_type) {
    auto input_model = load_model("model_editor/subgraph_extraction_tests.onnx");
    EXPECT_EQ(input_model->get_element_type(input_model->get_place_by_tensor_name("in1")), ov::element::f32);
    EXPECT_EQ(input_model->get_element_type(input_model->get_place_by_tensor_name("in2")), ov::element::f32);
    input_model->set_element_type(input_model->get_place_by_tensor_name("in3"), ov::element::f16);
    EXPECT_EQ(input_model->get_element_type(input_model->get_place_by_tensor_name("in3")), ov::element::f16);
    EXPECT_THROW(input_model->get_element_type(nullptr), ov::Exception);
}

OPENVINO_TEST(onnx_editor, subgraph__duplicated_output) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_ab_duplicated_output.onnx", &front_end);
    const auto y_out = input_model->get_place_by_tensor_name("Y");
    EXPECT_TRUE(y_out);
    input_model->extract_subgraph({}, {y_out});

    auto model = front_end->convert(input_model);
    const auto model_ref = convert_model("model_editor/add_ab.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, subgraph__duplicated_output_2) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    FrontEnd::Ptr front_end;
    auto input_model = load_model("model_editor/add_ab_duplicated_output.onnx", &front_end);
    const auto y_out_1 = input_model->get_place_by_tensor_name("Y");
    const auto y_out_2 = input_model->get_place_by_tensor_name("Y");
    EXPECT_TRUE(y_out_1);
    EXPECT_TRUE(y_out_2);
    input_model->extract_subgraph({}, {y_out_1, y_out_2});

    auto model = front_end->convert(input_model);
    const auto model_ref = convert_model("model_editor/add_ab_duplicated_output.onnx");

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CONSUMERS_COUNT);

    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

OPENVINO_TEST(onnx_editor, onnx_shape_infer_exception) {
    SKIP_ONNX_EDITOR_IF_GRAPH_ITERATOR_ENABLED();
    auto input_model = load_model("model_editor/onnx_shape_infer_exception.onnx");

    EXPECT_NO_THROW(input_model->extract_subgraph({input_model->get_place_by_operation_name("input_ReduceMin")}, {}));
}
