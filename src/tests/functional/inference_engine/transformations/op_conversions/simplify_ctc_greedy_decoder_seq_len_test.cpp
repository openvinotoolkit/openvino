// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>


#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 7 });
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i64, ngraph::Shape{ 1 });

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, true);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 7 });
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i64, ngraph::Shape{ 1 });

        element::Type data_type = data1->get_element_type();
        element::Type seq_len_type = seq_len1->get_element_type();
        element::Type ci_type = element::i32;
        element::Type sl_type = element::i32;
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(data1,
                                                                     ngraph::opset6::Constant::create(element::i32,
                                                                                                      Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(data1);
        auto axisT = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, indexT, axisT);

        auto axisN = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexN = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, indexN, axisN);

        auto start = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto step = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto plus1 = opset6::Constant::create(element::i64, Shape{1}, {1});
        auto plusT = std::make_shared<ngraph::opset6::Add>(T, plus1);
        auto const_plusT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto plusT_scalar = std::make_shared<ngraph::opset6::Squeeze>(plusT, const_plusT);
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, plusT_scalar, step, seq_len_type);

        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                OutputVector{T->output(0), N->output(0)}, 0);

        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                seq_len1, mask_shape->output(0));
        auto transpose_upper_bounds = std::make_shared<ngraph::opset6::Transpose>(upper_bounds->output(0),
                                                                                  ngraph::opset6::Constant::create(seq_len_type,
                                                                                                                   Shape({2}), {1, 0}));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(transpose_upper_bounds->output(0),
                                                                            range1T->output(0));

        auto mask_val_true = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto mask_val_false = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, mask_val_true, mask_val_false);
        auto transpose_seq_mask = std::make_shared<ngraph::opset6::Transpose>(seq_mask->output(0),
                                                                              ngraph::opset6::Constant::create(seq_len_type,
                                                                                                               Shape({2}), {1, 0}));
        auto transpose_seq_mask_f32 = std::make_shared<ngraph::opset6::Convert>(transpose_seq_mask->output(0), data_type);
        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                                     transpose_seq_mask_f32->output(0),
                                                                                     true);

        auto squeeze2_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ngraph::opset6::Squeeze>(simplified_decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ngraph::opset6::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        auto output_i = std::make_shared<ngraph::opset6::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = opset6::Constant::create(ci_type, Shape{}, {-1});
        auto where_equal_minus1 = std::make_shared<ngraph::opset6::Equal>(output_i, minus1);

        auto seq_mask_const0 = opset6::Constant::create(ci_type, Shape{1}, {0});
        auto seq_mask_const1 = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_mask = std::make_shared<ngraph::opset6::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_len = std::make_shared<ngraph::opset6::ReduceSum>(output_seq_mask, seq_mask_axis);
        auto output_seq_len_i = std::make_shared<ngraph::opset6::Convert>(output_seq_len->output(0), sl_type);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ output_i, output_seq_len_i }, ngraph::ParameterVector{ data1, seq_len1 });
    }
}

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenDynamicInputShapeTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic());
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{ 1 });

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, true, element::i64, element::i32);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic());
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{ 1 });

        element::Type data_type = data1->get_element_type();
        element::Type seq_len_type = seq_len1->get_element_type();
        element::Type ci_type = element::i64;
        element::Type sl_type = element::i32;
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(data1,
                                                                     ngraph::opset6::Constant::create(element::i32,
                                                                                                      Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(data1);
        auto axisT = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, indexT, axisT);

        auto axisN = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexN = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, indexN, axisN);

        auto start = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto step = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto plus1 = opset6::Constant::create(element::i64, Shape{1}, {1});
        auto plusT = std::make_shared<ngraph::opset6::Add>(T, plus1);
        auto const_plusT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto plusT_scalar = std::make_shared<ngraph::opset6::Squeeze>(plusT, const_plusT);
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, plusT_scalar, step, seq_len_type);

        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                OutputVector{T->output(0), N->output(0)}, 0);

        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                seq_len1, mask_shape->output(0));
        auto transpose_upper_bounds = std::make_shared<ngraph::opset6::Transpose>(upper_bounds->output(0),
                                                                                  ngraph::opset6::Constant::create(seq_len_type,
                                                                                                                   Shape({2}), {1, 0}));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(transpose_upper_bounds->output(0),
                                                                            range1T->output(0));

        auto mask_val_true = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto mask_val_false = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, mask_val_true, mask_val_false);
        auto transpose_seq_mask = std::make_shared<ngraph::opset6::Transpose>(seq_mask->output(0),
                                                                              ngraph::opset6::Constant::create(seq_len_type,
                                                                                                               Shape({2}), {1, 0}));
        auto transpose_seq_mask_f = std::make_shared<ngraph::opset6::Convert>(transpose_seq_mask->output(0), data_type);
        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                                     transpose_seq_mask_f->output(0),
                                                                                     true);

        auto squeeze2_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ngraph::opset6::Squeeze>(simplified_decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ngraph::opset6::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        auto output_i = std::make_shared<ngraph::opset6::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = opset6::Constant::create(ci_type, Shape{}, {-1});
        auto where_equal_minus1 = std::make_shared<ngraph::opset6::Equal>(output_i, minus1);

        auto seq_mask_const0 = opset6::Constant::create(ci_type, Shape{1}, {0});
        auto seq_mask_const1 = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_mask = std::make_shared<ngraph::opset6::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_len = std::make_shared<ngraph::opset6::ReduceSum>(output_seq_mask, seq_mask_axis);
        auto output_seq_len_i = std::make_shared<ngraph::opset6::Convert>(output_seq_len->output(0), sl_type);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ output_i, output_seq_len_i }, ngraph::ParameterVector{ data1, seq_len1 });
    }
}

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenDynamicBatchTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{Dimension::dynamic(), 3, 7});
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{Dimension::dynamic()});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, true, element::i32, element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{Dimension::dynamic(), 3, 7});
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{Dimension::dynamic()});

        element::Type data_type = data1->get_element_type();
        element::Type seq_len_type = seq_len1->get_element_type();
        element::Type ci_type = element::i32;
        element::Type sl_type = element::i64;
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(data1,
                                                                     ngraph::opset6::Constant::create(element::i32,
                                                                                                      Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(data1);
        auto axisT = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, indexT, axisT);

        auto axisN = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexN = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, indexN, axisN);

        auto start = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto step = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto plus1 = opset6::Constant::create(element::i64, Shape{1}, {1});
        auto plusT = std::make_shared<ngraph::opset6::Add>(T, plus1);
        auto const_plusT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto plusT_scalar = std::make_shared<ngraph::opset6::Squeeze>(plusT, const_plusT);
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, plusT_scalar, step, seq_len_type);

        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                OutputVector{T->output(0), N->output(0)}, 0);

        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                seq_len1, mask_shape->output(0));
        auto transpose_upper_bounds = std::make_shared<ngraph::opset6::Transpose>(upper_bounds->output(0),
                                                                                  ngraph::opset6::Constant::create(seq_len_type,
                                                                                                                   Shape({2}), {1, 0}));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(transpose_upper_bounds->output(0),
                                                                            range1T->output(0));

        auto mask_val_true = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto mask_val_false = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, mask_val_true, mask_val_false);
        auto transpose_seq_mask = std::make_shared<ngraph::opset6::Transpose>(seq_mask->output(0),
                                                                              ngraph::opset6::Constant::create(seq_len_type,
                                                                                                               Shape({2}), {1, 0}));
        auto transpose_seq_mask_f = std::make_shared<ngraph::opset6::Convert>(transpose_seq_mask->output(0), data_type);
        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                                     transpose_seq_mask_f->output(0),
                                                                                     true);

        auto squeeze2_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ngraph::opset6::Squeeze>(simplified_decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ngraph::opset6::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        auto output_i = std::make_shared<ngraph::opset6::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = opset6::Constant::create(ci_type, Shape{}, {-1});
        auto where_equal_minus1 = std::make_shared<ngraph::opset6::Equal>(output_i, minus1);

        auto seq_mask_const0 = opset6::Constant::create(ci_type, Shape{1}, {0});
        auto seq_mask_const1 = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_mask = std::make_shared<ngraph::opset6::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_len = std::make_shared<ngraph::opset6::ReduceSum>(output_seq_mask, seq_mask_axis);
        auto output_seq_len_i = std::make_shared<ngraph::opset6::Convert>(output_seq_len->output(0), sl_type);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ output_i, output_seq_len_i }, ngraph::ParameterVector{ data1, seq_len1 });
    }
}

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenDynamicSeqLenTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, true, ngraph::element::i64, ngraph::element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});

        element::Type data_type = data1->get_element_type();
        element::Type seq_len_type = seq_len1->get_element_type();
        element::Type ci_type = element::i64;
        element::Type sl_type = element::i64;
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(data1,
                                                                     ngraph::opset6::Constant::create(element::i32,
                                                                                                      Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(data1);
        auto axisT = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, indexT, axisT);

        auto axisN = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexN = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, indexN, axisN);

        auto start = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto step = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto plus1 = opset6::Constant::create(element::i64, Shape{1}, {1});
        auto plusT = std::make_shared<ngraph::opset6::Add>(T, plus1);
        auto const_plusT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto plusT_scalar = std::make_shared<ngraph::opset6::Squeeze>(plusT, const_plusT);
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, plusT_scalar, step, seq_len_type);

        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                OutputVector{T->output(0), N->output(0)}, 0);

        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                seq_len1, mask_shape->output(0));
        auto transpose_upper_bounds = std::make_shared<ngraph::opset6::Transpose>(upper_bounds->output(0),
                                                                                  ngraph::opset6::Constant::create(seq_len_type,
                                                                                                                   Shape({2}), {1, 0}));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(transpose_upper_bounds->output(0),
                                                                            range1T->output(0));

        auto mask_val_true = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto mask_val_false = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, mask_val_true, mask_val_false);
        auto transpose_seq_mask = std::make_shared<ngraph::opset6::Transpose>(seq_mask->output(0),
                                                                              ngraph::opset6::Constant::create(seq_len_type,
                                                                                                               Shape({2}), {1, 0}));
        auto transpose_seq_mask_f = std::make_shared<ngraph::opset6::Convert>(transpose_seq_mask->output(0), data_type);
        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                                     transpose_seq_mask_f->output(0),
                                                                                     true);

        auto squeeze2_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ngraph::opset6::Squeeze>(simplified_decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ngraph::opset6::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        auto output_i = std::make_shared<ngraph::opset6::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = opset6::Constant::create(ci_type, Shape{}, {-1});
        auto where_equal_minus1 = std::make_shared<ngraph::opset6::Equal>(output_i, minus1);

        auto seq_mask_const0 = opset6::Constant::create(ci_type, Shape{1}, {0});
        auto seq_mask_const1 = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_mask = std::make_shared<ngraph::opset6::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_len = std::make_shared<ngraph::opset6::ReduceSum>(output_seq_mask, seq_mask_axis);
        auto output_seq_len_i = std::make_shared<ngraph::opset6::Convert>(output_seq_len->output(0), sl_type);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ output_i, output_seq_len_i },
                                                   ngraph::ParameterVector{ data1, seq_len1 });
    }
}

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenWrongBlankIndexTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});
        auto blank_index = op::Constant::create(element::i32, Shape{}, {5});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blank_index,
                                                                                   true, ngraph::element::i64, ngraph::element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});
        auto blank_index1 = op::Constant::create(element::i32, Shape{}, {5});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data1, seq_len1, blank_index1,
                                                                                   true, ngraph::element::i64, ngraph::element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data1, seq_len1 });
    }
}

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenDynamicSeqLenWithBlankIndexTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});
        auto blank_index = op::Constant::create(element::i32, Shape{}, {6});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blank_index,
                                                                                   true, ngraph::element::i64, ngraph::element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});

        element::Type data_type = data1->get_element_type();
        element::Type seq_len_type = seq_len1->get_element_type();
        element::Type ci_type = element::i64;
        element::Type sl_type = element::i64;
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(data1,
                                                                     ngraph::opset6::Constant::create(element::i32,
                                                                                                      Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(data1);
        auto axisT = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, indexT, axisT);

        auto axisN = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto indexN = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, indexN, axisN);

        auto start = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto step = opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto plus1 = opset6::Constant::create(element::i64, Shape{1}, {1});
        auto plusT = std::make_shared<ngraph::opset6::Add>(T, plus1);
        auto const_plusT = ngraph::opset6::Constant::create(seq_len_type, Shape{1}, {0});
        auto plusT_scalar = std::make_shared<ngraph::opset6::Squeeze>(plusT, const_plusT);
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, plusT_scalar, step, seq_len_type);

        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                OutputVector{T->output(0), N->output(0)}, 0);

        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                seq_len1, mask_shape->output(0));
        auto transpose_upper_bounds = std::make_shared<ngraph::opset6::Transpose>(upper_bounds->output(0),
                                                                                  ngraph::opset6::Constant::create(seq_len_type,
                                                                                                                   Shape({2}), {1, 0}));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(transpose_upper_bounds->output(0),
                                                                            range1T->output(0));

        auto mask_val_true = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {1});
        auto mask_val_false = ngraph::opset6::Constant::create(seq_len_type, Shape{}, {0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, mask_val_true, mask_val_false);
        auto transpose_seq_mask = std::make_shared<ngraph::opset6::Transpose>(seq_mask->output(0),
                                                                              ngraph::opset6::Constant::create(seq_len_type,
                                                                                                               Shape({2}), {1, 0}));
        auto transpose_seq_mask_f = std::make_shared<ngraph::opset6::Convert>(transpose_seq_mask->output(0), data_type);
        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                                     transpose_seq_mask_f->output(0),
                                                                                     true);

        auto squeeze2_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ngraph::opset6::Squeeze>(simplified_decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ngraph::opset6::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        auto output_i = std::make_shared<ngraph::opset6::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = opset6::Constant::create(ci_type, Shape{}, {-1});
        auto where_equal_minus1 = std::make_shared<ngraph::opset6::Equal>(output_i, minus1);

        auto seq_mask_const0 = opset6::Constant::create(ci_type, Shape{1}, {0});
        auto seq_mask_const1 = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_mask = std::make_shared<ngraph::opset6::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = opset6::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_len = std::make_shared<ngraph::opset6::ReduceSum>(output_seq_mask, seq_mask_axis);
        auto output_seq_len_i = std::make_shared<ngraph::opset6::Convert>(output_seq_len->output(0), sl_type);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ output_i, output_seq_len_i },
                                                   ngraph::ParameterVector{ data1, seq_len1 });
    }
}

TEST_F(TransformationTestsF, SimplifyCTCGreedyDecoderSeqLenDynamicSeqLenParamWithBlankIndexTest) {
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});
        auto blank_index = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{1});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blank_index,
                                                                                   true, ngraph::element::i64, ngraph::element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data, seq_len, blank_index });

        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::PartialShape{2, Dimension::dynamic(), 7});
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{2});
        auto blank_index1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::PartialShape{1});

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data1, seq_len1, blank_index1,
                                                                                   true, ngraph::element::i64, ngraph::element::i64);
        auto res_1 = std::make_shared<opset6::Result>(decoder_v6->output(0));
        auto res_2 = std::make_shared<opset6::Result>(decoder_v6->output(1));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_1, res_2 }, ngraph::ParameterVector{ data1, seq_len1, blank_index1 });
    }
}
