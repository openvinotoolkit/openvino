// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "transformations/op_conversions/simplify_ctc_greedy_decoder.hpp"

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SimplifyCTCGreedyDecoder, "SimplifyCTCGreedyDecoder", 0);

ngraph::pass::SimplifyCTCGreedyDecoder::SimplifyCTCGreedyDecoder() {
    MATCHER_SCOPE(SimplifyCTCGreedyDecoder);
    auto decoder = pattern::wrap_type<opset6::CTCGreedyDecoderSeqLen>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto decoder_v6 = std::dynamic_pointer_cast<opset6::CTCGreedyDecoderSeqLen> (m.get_match_root());
        if (!decoder_v6) {
            return false;
        }
        element::Type seq_len_type = decoder_v6->input_value(1).get_element_type();
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(decoder_v6->input_value(0),
                                                                     ngraph::opset6::Constant::create(element::i32,
                                                               Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(decoder_v6->input_value(0));
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
                decoder_v6->input_value(1), mask_shape->output(0));
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
        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                                     transpose_seq_mask->output(0),
                                                                                     decoder_v6->get_merge_repeated());
        simplified_decoder->set_friendly_name(decoder_v6->get_friendly_name());

        auto squeeze2_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ngraph::opset6::Squeeze>(simplified_decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ngraph::opset6::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ngraph::opset6::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        element::Type ci_type = decoder_v6->get_classes_index_type();
        element::Type sl_type = decoder_v6->get_sequence_length_type();
        auto output_i = std::make_shared<ngraph::opset6::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = opset6::Constant::create(sl_type, Shape{}, {-1});
        auto where_equal_minus1 = std::make_shared<ngraph::opset6::Equal>(output_i, minus1);

        auto seq_mask_const0 = opset6::Constant::create(sl_type, Shape{1}, {0});
        auto seq_mask_const1 = opset6::Constant::create(sl_type, Shape{1}, {1});
        auto output_seq_mask = std::make_shared<ngraph::opset6::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = opset6::Constant::create(seq_len_type, Shape{1}, {1});
        auto output_seq_len = std::make_shared<ngraph::opset6::ReduceSum>(output_seq_mask, seq_mask_axis);
        ngraph::copy_runtime_info(decoder_v6, {transpose, simplified_decoder, data_shape, T, N, plusT, plusT_scalar, range1T, mask_shape, upper_bounds,
                                               squeeze2_output_f, squeeze1_output_f, transpose_upper_bounds, bool_seq_mask, seq_mask, transpose_seq_mask,
                                                   output_i, where_equal_minus1, output_seq_mask, output_seq_len});

        output_i->set_friendly_name(decoder_v6->get_friendly_name()+".0");
        output_seq_len->set_friendly_name(decoder_v6->get_friendly_name()+".1");
        ngraph::replace_node(decoder_v6, {output_i->output(0), output_seq_len->output(0)});

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(decoder, matcher_name);
    register_matcher(m, callback);
}
