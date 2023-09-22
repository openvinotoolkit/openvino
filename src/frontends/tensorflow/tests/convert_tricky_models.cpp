// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>

#include "common_test_utils/test_common.hpp"
#include "conversion_with_reference.hpp"
#include "gtest/gtest.h"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::frontend;
using namespace ov::frontend::tensorflow::tests;

namespace {
NamedOutputVector fake_translator_ragged_tensor_to_sparse(const NodeContext& node) {
    // NOTE: pay attention that this is a fake translator for RaggedTensorToSparse
    // only serves for testing purposes
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 1, "RaggedTensorToSparse expects at least two inputs.");
    auto node_name = node.get_name();
    auto row_splits = node.get_input(0);
    auto strings = node.get_input(1);

    // Override type of input tensor if this is a Parameter
    if (auto parameter = as_type_ptr<ov::opset10::Parameter>(strings.get_node_shared_ptr())) {
        parameter->set_partial_shape(ov::PartialShape{Dimension()});
        parameter->set_element_type(ov::element::u8);
        parameter->validate_and_infer_types();
    }

    row_splits = make_shared<ConvertLike>(row_splits, strings);
    auto const_one = make_shared<Constant>(row_splits.get_element_type(), Shape{}, 1);
    Output<Node> mul = make_shared<Multiply>(row_splits, const_one);
    auto const_two = make_shared<Constant>(ov::element::u8, Shape{}, 2);
    Output<Node> add = make_shared<Add>(strings, const_two);
    auto const_three = make_shared<Constant>(ov::element::u8, Shape{}, 3);
    Output<Node> sub = make_shared<Subtract>(strings, const_three);

    mul.get_tensor().add_names({node_name + ":0"});
    add.get_tensor().add_names({node_name + ":1"});
    sub.get_tensor().add_names({node_name + ":2"});

    return {{"sparse_indices", mul}, {"sparse_values", add}, {"sparse_dense_shape", sub}};
}
}  // namespace

TEST(FrontEndConvertTrickyModels, undefined_input_shape) {
    shared_ptr<Model> model;
    try {
        model = convert_model("undefined_input_shape/undefined_input_shape.pbtxt");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape::dynamic()));
        } else if (node->get_friendly_name() == "y") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "z") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape::dynamic()));
        }
    }
}

TEST(FrontEndConvertTrickyModels, simple_wide_and_deep) {
    shared_ptr<Model> model;
    try {
        model = convert_model("simple_wide_and_deep/simple_wide_and_deep.pbtxt");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    int num_emb_segment_sum = 0;
    for (auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<EmbeddingSegmentsSum>(node)) {
            ++num_emb_segment_sum;
        }
    }

    ASSERT_EQ(num_emb_segment_sum, 1) << "The number of EmbeddingSegmentsSum nodes must be 1";
}

TEST(FrontEndConvertTrickyModels, model_with_output_shapes) {
    shared_ptr<Model> model;
    try {
        model = convert_model("model_with_output_shapes_attr/model_with_output_shapes_attr.pbtxt");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "relu") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        }
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, AssertAndStringTensors) {
    {
        model = convert_model("string_tensors_model/string_tensors_model.pbtxt");
        // TODO: investigate - why we have redundant nodes after the conversion
        manager.register_pass<pass::MOCTransformations>(false);
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{2, 3});
        auto y = make_shared<Parameter>(f32, Shape{2, 3});
        auto cond = make_shared<Constant>(boolean, Shape{1, 1}, std::vector<bool>{true});
        auto select = make_shared<Select>(cond, x, y);

        model_ref = make_shared<Model>(OutputVector{select}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, UnsortedNodes) {
    { model = convert_model("forward_edge_model_unsorted/forward_edge_model_unsorted.pbtxt"); }
    { model_ref = convert_model("forward_edge_model/forward_edge_model.pbtxt"); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithSwishF32BodyGraph) {
    {
        model = convert_model("swish_f32/swish_f32.pbtxt");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{1, 112, 112, 32});
        auto const_add = make_shared<Constant>(f32, Shape{}, std::vector<float>{2});
        auto add = make_shared<Add>(x, const_add);
        auto sigmoid = make_shared<Sigmoid>(add);
        auto mul = make_shared<Multiply>(add, sigmoid);
        auto sigmoid2 = make_shared<Sigmoid>(mul);

        model_ref = make_shared<Model>(OutputVector{sigmoid2}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, PartitionedCall) {
    {
        model = convert_model("partitioned_call/partitioned_call.pbtxt");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        auto x = make_shared<Parameter>(i32, Shape{2});
        auto y = make_shared<Parameter>(i32, Shape{1});
        auto sub = make_shared<Subtract>(x, y);
        auto const_pow = make_shared<Constant>(i32, Shape{}, 2);
        auto pow = make_shared<Power>(sub, const_pow);

        model_ref = make_shared<Model>(OutputVector{pow}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithIf) {
    { model = convert_model("model_with_if/model_with_if.pbtxt"); }
    {
        // create then branch body graph
        auto then_x = make_shared<Parameter>(i32, Shape{2});
        auto then_y = make_shared<Parameter>(i32, Shape{1});
        auto add = make_shared<Add>(then_x, then_y);
        auto then_result = make_shared<Result>(add);
        auto then_model = make_shared<Model>(OutputVector{then_result}, ParameterVector{then_x, then_y});

        // create else branch body graph
        auto else_x = make_shared<Parameter>(i32, Shape{2});
        auto else_y = make_shared<Parameter>(i32, Shape{1});
        auto sub = make_shared<Subtract>(else_x, else_y);
        auto else_result = make_shared<Result>(sub);
        auto else_model = make_shared<Model>(OutputVector{else_result}, ParameterVector{else_x, else_y});

        // create the main graph
        auto x = make_shared<Parameter>(i32, Shape{2});
        auto y = make_shared<Parameter>(i32, Shape{1});
        auto cond_const = make_shared<Constant>(i32, Shape{}, 10);
        auto cond = make_shared<Greater>(x, cond_const);
        auto if_op = make_shared<If>(cond);
        if_op->set_then_body(then_model);
        if_op->set_else_body(else_model);
        if_op->set_input(x, then_x, else_x);
        if_op->set_input(y, then_y, else_y);
        if_op->set_output(then_result, else_result);

        model_ref = make_shared<Model>(OutputVector{if_op}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, InjectedBodyAndIf) {
    {
        model = convert_model("injected_body_and_if/injected_body_and_if.pbtxt");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        // create then branch body graph
        auto then_x = make_shared<Parameter>(i32, Shape{2});
        auto then_y = make_shared<Parameter>(i32, Shape{1});
        auto add = make_shared<Add>(then_x, then_y);
        auto then_result = make_shared<Result>(add);
        auto then_model = make_shared<Model>(OutputVector{then_result}, ParameterVector{then_x, then_y});

        // create else branch body graph
        auto else_x = make_shared<Parameter>(i32, Shape{2});
        auto else_y = make_shared<Parameter>(i32, Shape{1});
        auto sub = make_shared<Subtract>(else_x, else_y);
        auto pow_const = make_shared<Constant>(i32, Shape{}, 2);
        auto pow = make_shared<Power>(sub, pow_const);
        auto else_result = make_shared<Result>(pow);
        auto else_model = make_shared<Model>(OutputVector{else_result}, ParameterVector{else_x, else_y});

        // create the main graph
        auto x = make_shared<Parameter>(i32, Shape{2});
        auto y = make_shared<Parameter>(i32, Shape{1});
        auto cond_const = make_shared<Constant>(i32, Shape{}, 10);
        auto cond = make_shared<Greater>(x, cond_const);
        auto if_op = make_shared<If>(cond);
        if_op->set_then_body(then_model);
        if_op->set_else_body(else_model);
        if_op->set_input(x, then_x, else_x);
        if_op->set_input(y, then_y, else_y);
        if_op->set_output(then_result, else_result);

        model_ref = make_shared<Model>(OutputVector{if_op}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithDilatedGroupConvolution) {
    {
        model = convert_model("dilated_gconv_model/dilated_gconv_model.pbtxt");
        // need to call MOC to fuse BatchToSpace/SpaceToBatch with GroupConvolution
        manager.register_pass<pass::MOCTransformations>(false);
        // TODO: enable ATTRIBUTES, CONST_VALUES and ACCURACY checks, CVS-111900
        comparator.disable(FunctionsComparator::CmpValues::ATTRIBUTES);
        comparator.disable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{1, 129, 257, 384});
        auto transpose_before_const = make_shared<Constant>(i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        auto transpose_before = make_shared<Transpose>(x, transpose_before_const);
        auto const_filter = make_shared<Constant>(f32, Shape{384, 1, 1, 3, 3}, std::vector<float>(384 * 3 * 3, 0));
        Strides dilations{2, 2};
        CoordinateDiff pads_begin{2, 2};
        CoordinateDiff pads_end{2, 2};
        Strides strides{1, 1};
        auto gconv =
            make_shared<GroupConvolution>(transpose_before, const_filter, strides, pads_begin, pads_end, dilations);
        auto transpose_after_const = make_shared<Constant>(i64, Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
        auto transpose_after = make_shared<Transpose>(gconv, transpose_after_const);

        model_ref = make_shared<Model>(OutputVector{transpose_after}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithSaveV2) {
    {
        model = convert_model("model_savev2/model_savev2.pbtxt");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2});
        auto const_2 = make_shared<Constant>(element::f32, Shape{2}, vector<float>{1, 2});
        auto add = make_shared<Add>(x, const_2);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithConstResultSubgraphs) {
    { model = convert_model("model_with_const_result/model_with_const_result.pbtxt"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 60, 60, 1});
        auto perm_order = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 2});
        auto transpose_to_nchw = make_shared<Transpose>(x, perm_order);
        auto max_pool = make_shared<MaxPool>(transpose_to_nchw,
                                             Strides{2, 2},
                                             Strides{1, 1},
                                             Shape{0, 0},
                                             Shape{0, 0},
                                             Shape{2, 2},
                                             ov::op::RoundingType::FLOOR,
                                             ov::op::PadType::VALID,
                                             element::i64);
        auto inverse_order = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 2, 3, 1});
        auto transpose_to_nhwc = make_shared<Transpose>(max_pool, inverse_order);

        model_ref = make_shared<Model>(OutputVector{transpose_to_nhwc}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithIteratorGetNext) {
    { model = convert_model("model_with_iterator_get_next/model_with_iterator_get_next.pbtxt"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto y = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto sub = make_shared<Subtract>(x, y);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithQueueOperations) {
    { model = convert_model("model_with_queue_ops/model_with_queue_ops.pbtxt"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 160, 160, 3});
        auto y = make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 160, 160, 3});
        auto sub = make_shared<Subtract>(x, y);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithQueueOperations2) {
    { model = convert_model("model_with_queue_ops2/model_with_queue_ops2.pbtxt"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 3});
        auto y = make_shared<Constant>(element::f32,
                                       Shape{1, 1, 1, 3},
                                       vector<float>{123.68000030517578, 116.77899932861328, 103.93900299072266});
        auto sub = make_shared<Subtract>(x, y);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithLookupTableOperations) {
    { model = convert_model("model_with_lookup_table/model_with_lookup_table.pbtxt"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2});
        auto const_2 = make_shared<Constant>(element::f32, Shape{2}, vector<float>{1, 2});
        auto add = make_shared<Add>(x, const_2);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithIteratorGetNextAndUnsupportedOp) {
    { model = convert_model("unsupported_op_itergetnext/unsupported_op_itergetnext.pb"); }
    {
        // create then branch body graph
        auto x = make_shared<Parameter>(f32, Shape{2, 3});
        auto y = make_shared<Parameter>(f32, Shape{3});
        auto add = make_shared<Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithMultioutputBodyGraphNode) {
    {
        model = convert_model("partitioned_call2/partitioned_call2.pbtxt");
        // TODO: enable ATTRIBUTES check, CVS-111901
        comparator.disable(FunctionsComparator::CmpValues::ATTRIBUTES);
    }
    {
        auto x = make_shared<Parameter>(i32, Shape{5});
        auto y = make_shared<Parameter>(i32, Shape{5});
        auto sub = make_shared<Subtract>(x, y);
        auto const_three = make_shared<Constant>(i32, Shape{}, 3);
        auto const_ten = make_shared<Constant>(i32, Shape{}, 10);
        auto topk =
            make_shared<TopK>(sub, const_three, -1, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES, i32);
        auto add = make_shared<Add>(topk->output(1), const_ten);
        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithEmptyTensorListAndPushBack) {
    { model = convert_model("empty_tensor_list/empty_tensor_list.pb"); }
    {
        auto x = make_shared<Parameter>(f32, Shape{2, 3, 5});
        auto minus_one_const = make_shared<Constant>(i32, Shape{1}, -1);
        auto x_flatten = make_shared<Reshape>(x, minus_one_const, false);
        auto zero_const = make_shared<Constant>(i32, Shape{1}, 0);
        auto x_unsqueeze_flatten = make_shared<Unsqueeze>(x_flatten, zero_const);
        auto list_push_back = make_shared<Concat>(OutputVector{x_unsqueeze_flatten}, 0);
        auto list_push_back_shape = make_shared<ShapeOf>(list_push_back, element::i32);
        auto start = make_shared<Constant>(i32, Shape{1}, 0);
        auto stop = make_shared<Constant>(i32, Shape{1}, 1);
        auto step = make_shared<Constant>(i32, Shape{1}, 1);
        auto batch = make_shared<Slice>(list_push_back_shape, start, stop, step);
        auto shape_without_batch = make_shared<Constant>(i32, Shape{3}, vector<int32_t>{2, 3, 5});
        auto recover_item_shape = make_shared<Concat>(OutputVector{batch, shape_without_batch}, 0);
        auto recover_item = make_shared<Reshape>(list_push_back, recover_item_shape, false);
        model_ref = make_shared<Model>(OutputVector{recover_item}, ParameterVector{x});
    }
    comparator.disable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(FrontEndConversionWithReferenceTestsF, ModelWithAssertNode) {
    { model = convert_model("model_with_assert/model_with_assert.pb"); }
    {
        auto x = make_shared<Parameter>(i32, PartialShape{Dimension::dynamic()});
        auto y = make_shared<Parameter>(i32, PartialShape{Dimension::dynamic()});
        auto add = make_shared<Add>(x, y);
        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{x, y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, PartitionedCallWithUnique) {
    // This test aims to test named output ports for Unique operation
    { model = convert_model("partitioned_call_with_unique/partitioned_call_with_unique.pb"); }
    {
        auto x = make_shared<Parameter>(f32, Shape{5});
        auto relu = make_shared<Relu>(x);
        auto unique = make_shared<Unique>(relu, false, i32);
        auto const_one = make_shared<Constant>(i32, Shape{}, 1);
        auto add = make_shared<Add>(unique->output(2), const_one);
        auto sigmoid = make_shared<Sigmoid>(unique->output(0));
        model_ref = make_shared<Model>(OutputVector{sigmoid, add}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, RaggedTensorToSparse) {
    // This test aims to test named output ports for RaggedTensorToSparse operation
    // also, it tests propagation of custom type (specified in the extension) to Parameter node in the parent graph
    {
        // create FAKE conversion extension for RaggedTensorToSparse
        auto conv_ext = std::make_shared<ov::frontend::ConversionExtension>("RaggedTensorToSparse",
                                                                            fake_translator_ragged_tensor_to_sparse);
        model = convert_model("ragged_tensor_to_sparse/ragged_tensor_to_sparse.pb", conv_ext);
    }
    {
        auto strings = make_shared<Parameter>(u8, PartialShape{3});
        auto row_splits = make_shared<Parameter>(i32, PartialShape{5});
        auto convert_like = make_shared<ConvertLike>(row_splits, strings);

        auto const_one = make_shared<Constant>(u8, Shape{}, 1);
        Output<Node> mul = make_shared<Multiply>(convert_like, const_one);
        auto const_three = make_shared<Constant>(u8, Shape{}, 3);
        Output<Node> sub = make_shared<Subtract>(strings, const_three);

        auto target_shape1 = make_shared<Constant>(i32, Shape{1}, -1);
        auto reshape1 = make_shared<Reshape>(mul, target_shape1, false);
        auto target_shape2 = make_shared<Constant>(i32, Shape{1}, -1);
        auto reshape2 = make_shared<Reshape>(sub, target_shape2, false);

        auto concat = make_shared<Concat>(OutputVector{reshape1, reshape2}, 0);

        model_ref = make_shared<Model>(OutputVector{concat}, ParameterVector{row_splits, strings});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, MetaGraphVariables) {
    {
        model = convert_model("metagraph_variables/graph.meta");
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<Parameter>(element::f32, Shape{1});
        auto z = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{2, 2, 1, 1, 1, 2});
        auto add = make_shared<Add>(x, y);
        auto sub = make_shared<Subtract>(add, z);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, MetaGraphCut) {
    {
        model = convert_model("metagraph_variables/graph.meta", nullptr, {"y"});
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<Parameter>(element::f32, Shape{1});
        auto z = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{2, 2, 1, 1, 1, 2});
        auto add = make_shared<Add>(x, y);
        auto sub = make_shared<Subtract>(add, z);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, MetaGraphCutInputTensor) {
    {
        model = convert_model("metagraph_variables/graph.meta",
                              nullptr,
                              {"0:SubOperation"},
                              {ov::element::f32},
                              {Shape{2, 3}});
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto z = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{2, 2, 1, 1, 1, 2});
        auto sub = make_shared<Subtract>(x, z);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, MetaGraphCutOutputTensor) {
    {
        model = convert_model("metagraph_variables/graph.meta",
                              nullptr,
                              {"AddOperation:0"},
                              {ov::element::f32},
                              {Shape{2, 3}});
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto z = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{2, 2, 1, 1, 1, 2});
        auto sub = make_shared<Subtract>(x, z);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, MetaGraphCutIdentity) {
    {
        model = convert_model("metagraph_variables/graph.meta",
                              nullptr,
                              {"AddIdentity"},
                              {ov::element::f32},
                              {Shape{2, 3}});
        model->validate_nodes_and_infer_types();
    }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{2, 3});
        auto z = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{2, 2, 1, 1, 1, 2});
        auto sub = make_shared<Subtract>(x, z);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, MetaGraphMMAPCompare) {
    { model = convert_model("metagraph_variables/graph.meta"); }
    { model_ref = convert_model("metagraph_variables/graph.meta", nullptr, {}, {}, {}, {}, {}, true); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SplitInFunction) {
    {
        // create FAKE conversion extension for Split using named ports, this is not required for Split, but it tests
        // how named ports will work if there is one name and many outputs associated with it
        auto conv_ext = std::make_shared<ov::frontend::ConversionExtension>("Split", [](const NodeContext& node) {
            auto axis = node.get_input(0);
            auto value = node.get_input(1);
            auto num_split = node.get_attribute<int64_t>("num_split");

            auto split = make_shared<Split>(value, axis, num_split);
            NamedOutputVector res;
            for (const auto& output : split->outputs()) {
                res.push_back({"output", output});
            }
            return res;
        });
        model = convert_model("split_in_function/split_in_function.pbtxt", conv_ext);
    }
    {
        auto x = make_shared<Parameter>(f32, PartialShape{3, 20});

        auto const_zero = make_shared<Constant>(i32, Shape{}, 0);
        auto split = make_shared<Split>(x, const_zero, 3);
        auto add1 = make_shared<Add>(split->output(0), split->output(1));
        auto add2 = make_shared<Add>(add1, split->output(2));

        model_ref = make_shared<Model>(OutputVector{add2}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ResourceGatherModel) {
    // This test aims to check basic support of ResourceGather operation
    // and cutting an input model with specified shapes and types
    {
        model = convert_model("resource_gather_model/resource_gather_model.pbtxt",
                              nullptr,
                              {"1:embedding_lookup1", "1:embedding_lookup2"},
                              {element::i32, element::i32},
                              {Shape{7, 2}, Shape{3}});
    }
    {
        auto ind1 = make_shared<Parameter>(i32, Shape{7, 2});
        auto table1 = make_shared<Constant>(f32, Shape{2, 3}, vector<float>{1, 2, 3, 4, 5, 6});
        auto axis1 = make_shared<Constant>(i64, Shape{}, 0);

        auto ind2 = make_shared<Parameter>(i32, Shape{3});
        auto table2 = make_shared<Constant>(f32, Shape{5}, vector<float>{10, 11, 12, 13, 14});
        auto axis2 = make_shared<Constant>(i64, Shape{}, 0);

        auto gather1 = make_shared<Gather>(table1, ind1, axis1);
        auto gather2 = make_shared<Gather>(table2, ind2, axis2);

        auto mul = make_shared<Multiply>(gather1, gather2);

        model_ref = make_shared<Model>(OutputVector{mul}, ParameterVector{ind1, ind2});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, NonMaxSuppressionWithNamedOutputs) {
    // The purpose of this test is to check that named output ports of TensorFlow NMS operation are connected correctly
    // to its consumers
    { model = convert_model("nms_named_outputs/nms_named_outputs.pb"); }
    {
        // prepare the first input for NMS
        auto boxes = make_shared<Parameter>(f32, PartialShape{2, 4});
        auto const_zero = make_shared<Constant>(i32, Shape{1}, 0);
        auto unsqueeze = make_shared<Unsqueeze>(boxes, const_zero);

        // prepare the second input for NMS
        auto scores = make_shared<Parameter>(f32, PartialShape{2});
        auto const_one_zero = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{0, 1});
        auto unsqueeze_2 = make_shared<Unsqueeze>(scores, const_one_zero);

        // create NMS node
        auto max_output_size = make_shared<Constant>(i32, Shape{}, 50);
        auto iou_threshold = make_shared<Constant>(f32, Shape{}, 0.4f);
        auto score_threshold = make_shared<Constant>(f32, Shape{}, 0.3f);
        auto soft_nms_sigma = make_shared<Constant>(f32, Shape{}, 0.1f);
        auto nms = make_shared<NonMaxSuppression>(unsqueeze,
                                                  unsqueeze_2,
                                                  max_output_size,
                                                  iou_threshold,
                                                  score_threshold,
                                                  soft_nms_sigma,
                                                  NonMaxSuppression::BoxEncodingType::CORNER,
                                                  false,
                                                  i32);

        // compute the first output - selected_indices
        auto slice_const_one = make_shared<Constant>(i32, Shape{1}, 1);
        auto slice_const_one_2 = make_shared<Constant>(i32, Shape{1}, 1);
        auto slice_const_two = make_shared<Constant>(i32, Shape{1}, 2);
        auto slice_const_three = make_shared<Constant>(i32, Shape{1}, 3);
        auto slice =
            make_shared<Slice>(nms->output(0), slice_const_two, slice_const_three, slice_const_one, slice_const_one_2);
        Output<Node> selected_indices = make_shared<Squeeze>(slice, slice_const_one_2);

        // compute the second output - selected_scores
        auto slice2_const_one = make_shared<Constant>(i32, Shape{1}, 1);
        auto slice2_const_one_2 = make_shared<Constant>(i32, Shape{1}, 1);
        auto slice2_const_two = make_shared<Constant>(i32, Shape{1}, 2);
        auto slice2_const_three = make_shared<Constant>(i32, Shape{1}, 3);
        auto slice2 =
            make_shared<Slice>(nms->output(1), slice_const_two, slice_const_three, slice_const_one, slice_const_one_2);
        Output<Node> selected_scores = make_shared<Squeeze>(slice2, slice_const_one_2);
        selected_scores = make_shared<ConvertLike>(selected_scores, boxes);
        selected_scores = make_shared<Convert>(selected_scores, i32);

        // compute the third output - valid_outputs
        auto squeeze_axes = make_shared<Constant>(i64, Shape{1}, 0);
        Output<Node> valid_outputs = make_shared<Squeeze>(nms->output(2), squeeze_axes);

        // make post-processing before the concatenation
        auto const_minus_one = make_shared<Constant>(i32, Shape{1}, -1);
        selected_indices = make_shared<Reshape>(selected_indices, const_minus_one, false);
        auto const_minus_one_2 = make_shared<Constant>(i32, Shape{1}, -1);
        selected_scores = make_shared<Reshape>(selected_scores, const_minus_one_2, false);
        auto const_minus_one_3 = make_shared<Constant>(i32, Shape{1}, -1);
        valid_outputs = make_shared<Reshape>(valid_outputs, const_minus_one_3, false);

        // concatenate all outputs in order to have the single output
        auto concat = make_shared<Concat>(OutputVector{selected_indices, selected_scores, valid_outputs}, 0);

        model_ref = make_shared<Model>(OutputVector{concat}, ParameterVector{boxes, scores});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, PartitionedCallsWithConvInBodyGraphs) {
    // The test aims to check that the conversion for the body graphs is performed with set input shapes
    // that allows to get more optimized ov::Model for the body graphs.
    // In particular, we check that the resulted graph contains Convolution operations instead of GroupConvolution
    { model = convert_model("partitioned_call_with_conv/partitioned_call_with_conv.pb"); }
    {
        auto input1 = make_shared<Parameter>(f32, Shape{1, 1, 10, 10});
        auto filter = make_shared<Parameter>(f32, Shape{3, 3, 1, 1});

        auto transpose_order = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{3, 2, 0, 1});

        auto tr_filter = make_shared<Transpose>(filter, transpose_order);

        auto conv = make_shared<Convolution>(input1,
                                             tr_filter,
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1},
                                             op::PadType::SAME_UPPER);

        model_ref = make_shared<Model>(OutputVector{conv}, ParameterVector{input1, filter});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ControlDependencyNumberOutputs) {
    // The test aims to check a number of outputs of the resulted model
    // If the node has dependent nodes by conditional edge, it is not terminating
    // and it should not go to the Result node
    { model = convert_model("control_dependency/control_dependency.pb"); }
    {
        auto input1 = make_shared<Parameter>(f32, Shape{2, 3});
        auto input2 = make_shared<Parameter>(f32, Shape{2, 3});

        // AddV2 node is excluded since it is not terminating
        auto sub = make_shared<Subtract>(input1, input2);

        model_ref = make_shared<Model>(OutputVector{sub}, ParameterVector{input1, input2});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, TF1IfWithNonExistentOpInBranch) {
    // This test aims to check conversion of a model with TF1 If operation that
    // contains unsupported operation in one branch
    // the conversion must avoid such branch in case proper condition freezing
    {
        bool cond_value = false;
        model = convert_model("tf1_if_with_nonexistent_op/tf1_if_with_nonexistent_op.pb",
                              nullptr,
                              {},
                              {},
                              {},
                              {"cond"},
                              {&cond_value});
    }
    {
        auto y = make_shared<Parameter>(f32, Shape{2, 3});
        auto ind = make_shared<Parameter>(i32, Shape{3});

        auto const_two = make_shared<Constant>(i32, Shape{}, 2);
        auto sub = make_shared<Subtract>(ind, const_two);

        auto convert = make_shared<Convert>(sub, f32);
        auto mul = make_shared<Multiply>(convert, y);

        model_ref = make_shared<Model>(OutputVector{mul}, ParameterVector{y, ind});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, ConvolutionWithDynamicInputChannel) {
    // This test aims to check conversion of a model with convolution of dynamic input channel
    // Namely, the resulted model must contain the regular convolution, not grouped convolution
    { model = convert_model("conv_with_dynamic_input_channel"); }
    {
        auto input = make_shared<Parameter>(f32, PartialShape{Dimension::dynamic(), 10, 10, Dimension::dynamic()});

        auto transpose_order = make_shared<Constant>(i64, Shape{4}, vector<int32_t>{0, 3, 1, 2});
        auto transpose = make_shared<Transpose>(input, transpose_order);

        auto filter = make_shared<Constant>(element::f32, Shape{6, 6, 3, 3}, vector<float>(6 * 6 * 3 * 3, 0.0f));
        auto conv = make_shared<Convolution>(transpose,
                                             filter,
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1},
                                             op::PadType::SAME_UPPER);

        auto transpose_order_back = make_shared<Constant>(i64, Shape{4}, vector<int32_t>{0, 2, 3, 1});
        auto transpose_back = make_shared<Transpose>(conv, transpose_order_back);

        model_ref = make_shared<Model>(OutputVector{transpose_back}, ParameterVector{input});
    }
}
