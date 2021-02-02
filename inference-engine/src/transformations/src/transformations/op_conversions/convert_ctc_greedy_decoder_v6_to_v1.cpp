// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "transformations/op_conversions/convert_ctc_greedy_decoder_v6_to_v1.hpp"

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertCTCGreedyDecoderV6ToV1, "ConvertCTCGreedyDecoderV6ToV1", 0);

ngraph::pass::ConvertCTCGreedyDecoderV6ToV1::ConvertCTCGreedyDecoderV6ToV1() {
    MATCHER_SCOPE(ConvertCTCGreedyDecoderV6ToV1);
    auto decoder = pattern::wrap_type<opset6::CTCGreedyDecoderSeqLen>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto & pattern_value = m.get_pattern_value_map();
        const auto & m_decoder = pattern_value.at(decoder);
        auto decoder_v6 = std::dynamic_pointer_cast<ngraph::opset6::CTCGreedyDecoderSeqLen> (m_decoder.get_node_shared_ptr());
        if (!decoder_v6) {
            return false;
        }

        auto transpose = std::make_shared<ngraph::opset6::Transpose>(decoder_v6->input_value(0), ngraph::opset6::Constant::create(element::i64,
                                                               Shape({3}), {1, 0, 2}));
        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(decoder_v6->input_value(0));
        auto constT_0 = ngraph::opset6::Constant::create(element::i64, Shape{}, {-1});
        auto constT_1 = ngraph::opset6::Constant::create(element::i64, Shape{}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, constT_1, constT_0);

        auto constN_0 = ngraph::opset6::Constant::create(element::i64, Shape{}, {0});
        auto constN_1 = ngraph::opset6::Constant::create(element::i64, Shape{}, {-1});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, constN_0, constN_0);

        auto start = opset6::Constant::create(element::i64, Shape{}, std::vector<int64_t >({1}));
        auto step = opset6::Constant::create(element::i64, Shape{}, std::vector<int64_t >({1}));
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, T, step,
                                                               decoder_v6->input(1).get_element_type());

        auto constUnsqueeze1 = ngraph::opset6::Constant::create(element::i64, Shape{}, {0});
        auto tT = std::make_shared<ngraph::opset6::Unsqueeze>(T, constUnsqueeze1);
        auto constUnsqueeze2 = ngraph::opset6::Constant::create(element::i64, Shape{}, {0});
        auto tN = std::make_shared<ngraph::opset6::Unsqueeze>(N, constUnsqueeze2);
        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                OutputVector{tT->output(0), tN->output(0)}, 0);
        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                decoder_v6->input_value(1), mask_shape->output(0));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(upper_bounds->output(0),
                                                                            range1T->output(0));
        auto const_0f = ngraph::opset6::Constant::create(element::f64, Shape{}, {0.0});
        auto const_1f = ngraph::opset6::Constant::create(element::f64, Shape{}, {1.0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, const_1f, const_0f);

        auto simplified_decoder = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(transpose,
                                                                             seq_mask->output(0),
                                                                             decoder_v6->get_merge_repeated());
        simplified_decoder->set_friendly_name(decoder_v6->get_friendly_name());
        ngraph::copy_runtime_info(decoder_v6, {simplified_decoder, data_shape, T, N, range1T, tT, tN, mask_shape, upper_bounds, bool_seq_mask, seq_mask});
        decoder_v6->output(0).replace(simplified_decoder->output(0));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(decoder, matcher_name);
    register_matcher(m, callback);
}
