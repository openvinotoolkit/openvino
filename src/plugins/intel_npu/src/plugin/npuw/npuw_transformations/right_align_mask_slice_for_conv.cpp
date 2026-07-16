// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "right_align_mask_slice_for_conv.hpp"

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class RightAlignMaskSliceForConvImpl : public ov::pass::MatcherPass {
public:
    // In LFM2-like models, where `attention_mask` is also consumed by the Conv subgraph,
    // the Slice operation is inserted to extract the current tokens from the `attention_mask`.
    // The mask is sliced from the left side, what will be an issue in the chunked
    // prefill's first iteration, as left part of that mask will be filled with zeroes
    // (due to the right padding of the input_ids in NPUW prefill).
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::RightAlignMaskSliceForConvImpl");
    explicit RightAlignMaskSliceForConvImpl() {
        auto attention_mask = opp::wrap_type<ov::op::v0::Parameter>();
        auto attention_mask_slice = opp::wrap_type<ov::op::v8::Slice>({attention_mask, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({attention_mask_slice, opp::any_input()});
        auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({convert, opp::any_input()});
        auto add = opp::wrap_type<ov::op::v1::Add>({multiply, opp::any_input()});
        auto multiply_with_embedd = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), add});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({multiply_with_embedd, opp::any_input()});
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
        // VariadicSplit into B,C,~h in LFM2.
        auto variadic_split = opp::wrap_type<ov::op::v1::VariadicSplit>({transpose, opp::any_input(), opp::any_input()});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_attention_mask = node_to_output.at(attention_mask).get_node_shared_ptr();
            if (matched_attention_mask->output(0).get_names().count(std::string(ov::npuw::attention_mask_name)) == 0) {
                return false;
            }
            auto matched_attention_mask_slice = node_to_output.at(attention_mask_slice).get_node_shared_ptr();
            auto matched_attention_mask_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();

            auto const_zero = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
            auto const_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 1);
            auto mask_shape_of = std::make_shared<ov::op::v3::ShapeOf>(matched_attention_mask);
            auto mask_len = std::make_shared<ov::op::v8::Gather>(mask_shape_of, const_one, const_zero);
            auto mask_len_rank_one = std::make_shared<ov::op::v0::Unsqueeze>(mask_len, const_zero);
            // 2nd argument to Slice is stop point:
            auto current_len = matched_attention_mask_slice->input(2).get_source_output().get_node_shared_ptr();
            auto current_offset = std::make_shared<ov::op::v1::Subtract>(mask_len, current_len);

            auto const_one_rank_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
            auto new_slice = std::make_shared<ov::op::v8::Slice>(matched_attention_mask, current_offset,
                mask_len_rank_one, const_one_rank_one, const_one_rank_one);
            matched_attention_mask_unsqueeze->input(0).replace_source_output(new_slice);
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(variadic_split, "RightAlignMaskSliceForConvImpl"),
                         std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
}  // anonymous namespace

bool ov::npuw::RightAlignMaskSliceForConv::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager("right-align-mask-slice-for-conv");
    manager.set_per_pass_validation(true);
    manager.register_pass<RightAlignMaskSliceForConvImpl>();
    manager.run_passes(model);
    return true;
}
