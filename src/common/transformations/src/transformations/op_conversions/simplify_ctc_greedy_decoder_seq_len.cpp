// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/ctc_greedy_decoder.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::SimplifyCTCGreedyDecoderSeqLen::SimplifyCTCGreedyDecoderSeqLen() {
    MATCHER_SCOPE(SimplifyCTCGreedyDecoderSeqLen);
    auto decoder = pattern::wrap_type<ov::op::v6::CTCGreedyDecoderSeqLen>();

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto decoder_seq_len = ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(m.get_match_root());
        if (!decoder_seq_len) {
            return false;
        }

        if (decoder_seq_len->get_input_size() > 2) {
            const auto data_pshape = decoder_seq_len->get_input_partial_shape(0);
            auto blank_index =
                ov::as_type_ptr<ov::op::v0::Constant>(decoder_seq_len->input_value(2).get_node_shared_ptr());
            if (!blank_index || data_pshape.rank().is_dynamic() || data_pshape[2].is_dynamic()) {
                return false;
            }

            const std::vector<int64_t>& blank_index_values = blank_index->cast_vector<int64_t>();
            const auto num_classes = decoder_seq_len->get_input_partial_shape(0)[2].get_length();
            if (blank_index_values[0] != (num_classes - 1)) {
                return false;
            }
        }

        element::Type data_type = decoder_seq_len->input_value(0).get_element_type();
        element::Type seq_len_type = decoder_seq_len->input_value(1).get_element_type();
        // Transposing input data channels from [N, T, C] to [T, N, C]. Need for compatible with CTCGreedyDecoder v1
        auto transpose =
            std::make_shared<ov::op::v1::Transpose>(decoder_seq_len->input_value(0),
                                                    ov::op::v0::Constant::create(element::i32, Shape({3}), {1, 0, 2}));
        // Receive time and batch dimensions and concatenate to [T, N] tensor shapes
        auto data_shape = std::make_shared<ov::op::v3::ShapeOf>(decoder_seq_len->input_value(0));
        auto axisT = ov::op::v0::Constant::create(seq_len_type, Shape{}, {0});
        auto indexT = ov::op::v0::Constant::create(seq_len_type, Shape{1}, {1});
        auto T = std::make_shared<ov::op::v1::Gather>(data_shape, indexT, axisT);

        auto axisN = ov::op::v0::Constant::create(seq_len_type, Shape{}, {0});
        auto indexN = ov::op::v0::Constant::create(seq_len_type, Shape{1}, {0});
        auto N = std::make_shared<ov::op::v1::Gather>(data_shape, indexN, axisN);

        auto start = ov::op::v0::Constant::create(seq_len_type, Shape{}, {1});
        auto step = ov::op::v0::Constant::create(seq_len_type, Shape{}, {1});
        auto plus1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto plusT = std::make_shared<ov::op::v1::Add>(T, plus1);
        auto const_plusT = ov::op::v0::Constant::create(seq_len_type, Shape{1}, {0});
        auto plusT_scalar = std::make_shared<ov::op::v0::Squeeze>(plusT, const_plusT);
        auto range1T = std::make_shared<ov::op::v4::Range>(start, plusT_scalar, step, seq_len_type);

        auto mask_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{T->output(0), N->output(0)}, 0);

        // Generate 2D tensor [T, N] for seq mask
        auto upper_bounds =
            std::make_shared<ov::op::v3::Broadcast>(decoder_seq_len->input_value(1), mask_shape->output(0));
        auto transpose_upper_bounds =
            std::make_shared<ov::op::v1::Transpose>(upper_bounds->output(0),
                                                    ov::op::v0::Constant::create(seq_len_type, Shape({2}), {1, 0}));
        // Compute boolean sequence mask
        auto bool_seq_mask =
            std::make_shared<ov::op::v1::GreaterEqual>(transpose_upper_bounds->output(0), range1T->output(0));

        // Generate resulted seq mask
        auto mask_val_true = ov::op::v0::Constant::create(seq_len_type, Shape{}, {1});
        auto mask_val_false = ov::op::v0::Constant::create(seq_len_type, Shape{}, {0});
        auto seq_mask = std::make_shared<ov::op::v1::Select>(bool_seq_mask, mask_val_true, mask_val_false);
        auto transpose_seq_mask =
            std::make_shared<ov::op::v1::Transpose>(seq_mask->output(0),
                                                    ov::op::v0::Constant::create(seq_len_type, Shape({2}), {1, 0}));
        auto transpose_seq_mask_f = std::make_shared<ov::op::v0::Convert>(transpose_seq_mask->output(0), data_type);
        // Create CTCGreedyDecoder with original merge_repeated attribute and connect data and resulted seq_mask
        auto decoder = std::make_shared<ov::op::v0::CTCGreedyDecoder>(transpose,
                                                                      transpose_seq_mask_f->output(0),
                                                                      decoder_seq_len->get_merge_repeated());
        decoder->set_friendly_name(decoder_seq_len->get_friendly_name());

        // Normalize output from CTCGreedyDecoder = output_f and create second output with output_seq_len
        auto squeeze2_axis = ov::op::v0::Constant::create(seq_len_type, Shape({1}), {3});
        auto squeeze2_output_f = std::make_shared<ov::op::v0::Squeeze>(decoder->output(0), squeeze2_axis);
        auto squeeze1_axis = ov::op::v0::Constant::create(seq_len_type, Shape({1}), {2});
        auto squeeze1_output_f = std::make_shared<ov::op::v0::Squeeze>(squeeze2_output_f->output(0), squeeze1_axis);

        element::Type ci_type = decoder_seq_len->get_classes_index_type();
        element::Type sl_type = decoder_seq_len->get_sequence_length_type();
        // CTCGreedyDecoder return floating point output. For Normalize output we need to convert output to
        // classes_index_type Receive the first output with correct classes_index_type
        auto output_i = std::make_shared<ov::op::v0::Convert>(squeeze1_output_f->output(0), ci_type);
        auto minus1 = ov::op::v0::Constant::create(ci_type, Shape{}, {-1});
        // Get to know where equal -1
        auto where_equal_minus1 = std::make_shared<ov::op::v1::Equal>(output_i, minus1);

        // Compute output seq mask
        auto seq_mask_const0 = ov::op::v0::Constant::create(ci_type, Shape{1}, {0});
        auto seq_mask_const1 = ov::op::v0::Constant::create(ci_type, Shape{1}, {1});
        auto output_seq_mask =
            std::make_shared<ov::op::v1::Select>(where_equal_minus1, seq_mask_const0, seq_mask_const1);
        auto seq_mask_axis = ov::op::v0::Constant::create(ci_type, Shape{1}, {1});
        // Receive the second output
        auto output_seq_len = std::make_shared<ov::op::v1::ReduceSum>(output_seq_mask, seq_mask_axis);
        // Receive the second output with correct seq_len_type
        auto output_seq_len_i = std::make_shared<ov::op::v0::Convert>(output_seq_len->output(0), sl_type);
        ov::copy_runtime_info(decoder_seq_len,
                              {transpose,
                               decoder,
                               data_shape,
                               T,
                               N,
                               plusT,
                               plusT_scalar,
                               range1T,
                               mask_shape,
                               upper_bounds,
                               squeeze2_output_f,
                               squeeze1_output_f,
                               transpose_upper_bounds,
                               bool_seq_mask,
                               seq_mask,
                               transpose_seq_mask,
                               transpose_seq_mask_f,
                               output_i,
                               where_equal_minus1,
                               output_seq_mask,
                               output_seq_len,
                               output_seq_len_i});

        output_i->set_friendly_name(decoder_seq_len->get_friendly_name() + ".0");
        output_seq_len_i->set_friendly_name(decoder_seq_len->get_friendly_name() + ".1");
        ov::replace_node(decoder_seq_len, {output_i->output(0), output_seq_len_i->output(0)});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(decoder, matcher_name);
    register_matcher(m, callback);
}
