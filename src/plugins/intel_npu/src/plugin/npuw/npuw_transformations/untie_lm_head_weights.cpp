// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "untie_lm_head_weights.hpp"

#include <algorithm>
#include <memory>

#include "cut_lm_head.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"

namespace opp = ov::pass::pattern;

namespace {

// True when an op of the decompression also feeds something else.
bool is_shared(const ov::Output<ov::Node>& out) {
    if (ov::op::util::is_constant(out.get_node())) {
        return false;
    }
    if (out.get_target_inputs().size() > 1) {
        return true;
    }
    const auto inputs = out.get_node()->input_values();
    return std::any_of(inputs.begin(), inputs.end(), is_shared);
}

// Copies the decompression ops; the constants they read are reused.
ov::Output<ov::Node> copy_decompression(const ov::Output<ov::Node>& out) {
    const auto node = out.get_node_shared_ptr();
    if (ov::op::util::is_constant(node)) {
        return out;
    }
    ov::OutputVector inputs;
    for (const auto& input : node->input_values()) {
        inputs.push_back(copy_decompression(input));
    }
    const auto copy = node->clone_with_new_inputs(inputs);
    copy->set_friendly_name(node->get_friendly_name() + "/lm_head");
    ov::copy_runtime_info(node, copy);
    return copy->output(out.get_index());
}

}  // namespace

ov::npuw::UntieLMHeadWeights::UntieLMHeadWeights() {
    const auto weights = std::make_shared<opp::op::CompressedWeightsBlock>(
        std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4},
        std::set<std::size_t>{2});
    const auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), weights});
    const ov::npuw::patterns::LMHeadTerminal terminal(matmul);
    const auto result = opp::wrap_type<ov::op::v0::Result>({terminal.output});

    ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
        const auto lm_head = m.get_pattern_value_map().at(matmul).get_node_shared_ptr();
        const auto table = lm_head->input_value(1);
        if (!is_shared(table)) {
            return false;
        }
        lm_head->input(1).replace_source_output(copy_decompression(table));
        return true;
    };
    register_matcher(std::make_shared<opp::Matcher>(result, "UntieLMHeadWeights"), callback);
}
