// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_vocab_sub128.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class InsertVocabSub128Matcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::InsertVocabSub128Matcher");

    InsertVocabSub128Matcher() {
        const auto qweight = opp::wrap_type<ov::op::v0::Constant>();
        const auto qzerop = opp::wrap_type<ov::op::v0::Constant>();
        const auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();
        const auto qweight_convert = opp::wrap_type<ov::op::v0::Convert>({qweight});
        const auto qzerop_convert = opp::wrap_type<ov::op::v0::Convert>({qzerop});
        const auto qsub = opp::wrap_type<ov::op::v1::Subtract>({qweight_convert, qzerop_convert});
        const auto qscale = opp::wrap_type<ov::op::v1::Multiply>({qsub, qcoeff});
        const auto qconvert = opp::optional<ov::op::v0::Convert>({qscale});
        const auto hidden = opp::any_input();
        const auto matmul = opp::wrap_type<ov::op::v0::MatMul>({hidden, qconvert});

        // Keep this terminal pattern aligned with CutLMHead: only vocabulary MatMuls
        // that form an LM-head result are eligible for Sub128 insertion.
        const auto matmul_add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
        const auto matmul_transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
        const auto matmul_convert = opp::wrap_type<ov::op::v0::Convert>({matmul});
        const auto div = opp::wrap_type<ov::op::v1::Multiply, ov::op::v1::Divide>({matmul, opp::any_input()});
        const auto tanh = opp::wrap_type<ov::op::v0::Tanh>({div});
        const auto matmul_multiply = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});
        const auto lm_head_output = std::make_shared<opp::op::Or>(ov::OutputVector{matmul->output(0),
                                                                                   matmul_add->output(0),
                                                                                   matmul_transpose->output(0),
                                                                                   matmul_convert->output(0),
                                                                                   matmul_multiply->output(0)});
        const auto result = opp::wrap_type<ov::op::v0::Result>({lm_head_output->output(0)});

        auto callback = [=](opp::Matcher& matcher) {
            const auto& values = matcher.get_pattern_value_map();
            const auto weight = values.at(qweight).get_node_shared_ptr();
            const auto zerop = values.at(qzerop).get_node_shared_ptr();
            const auto scale = values.at(qcoeff).get_node_shared_ptr();
            const auto subtract = values.at(qsub).get_node_shared_ptr();
            const auto matched_matmul =
                std::static_pointer_cast<ov::op::v0::MatMul>(values.at(matmul).get_node_shared_ptr());
            const auto matched_result =
                std::static_pointer_cast<ov::op::v0::Result>(values.at(result).get_node_shared_ptr());

            if (matched_result->get_rt_info().count("manually_added_output")) {
                return false;
            }

            const auto scale_shape = std::static_pointer_cast<ov::op::v0::Constant>(scale)->get_shape();
            const bool standard_layout = scale_shape.size() == 2 && scale_shape[1] == 1 &&
                                         !matched_matmul->get_transpose_a() && matched_matmul->get_transpose_b();
            const bool pretransposed_layout = scale_shape.size() == 2 && scale_shape[0] == 1 &&
                                              !matched_matmul->get_transpose_a() && !matched_matmul->get_transpose_b();

            if (weight->get_element_type() != ov::element::u8 || zerop->get_element_type() != ov::element::u8 ||
                weight->get_shape().size() != 2 || (!standard_layout && !pretransposed_layout)) {
                return false;
            }

            const auto weight_convert = subtract->input_value(0);
            const auto zerop_convert = subtract->input_value(1);
            const auto compute_type = weight_convert.get_element_type();
            const auto shift = ov::op::v0::Constant::create(compute_type, ov::Shape{}, {128});
            // PPP may not recognize the DQ subgraph after Sub128 insertion. Mark the source
            // converts as decompression to prevent PPP constant folding from materializing
            // enormous vocabulary tensors during KV-cache precision conversion.
            ov::mark_as_decompression(weight_convert.get_node_shared_ptr());
            ov::mark_as_decompression(zerop_convert.get_node_shared_ptr());
            const auto shifted_weight = std::make_shared<ov::op::v1::Subtract>(weight_convert, shift);
            const auto shifted_zerop = std::make_shared<ov::op::v1::Subtract>(zerop_convert, shift);
            shifted_weight->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
            shifted_zerop->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
            subtract->input(0).replace_source_output(shifted_weight);
            subtract->input(1).replace_source_output(shifted_zerop);
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(result, "InsertVocabSub128"), std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

ov::npuw::InsertVocabSub128::InsertVocabSub128() {
    add_matcher<InsertVocabSub128Matcher>();
}
