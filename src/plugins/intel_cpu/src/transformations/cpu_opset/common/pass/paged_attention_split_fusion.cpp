// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_split_fusion.hpp"

#include <utils/general_utils.h>

#include <openvino/core/rt_info.hpp>
#include "openvino/opsets/opset1.hpp"
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/utils/gen_pattern.hpp>

#include "itt.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "openvino/op/paged_attention.hpp"
#include "transformations/cpu_opset/common/op/paged_attention_split.hpp"
using namespace ov::gen_pattern;

namespace ov {
namespace intel_cpu {

PagedAttentionFusion::PagedAttentionFusion() {
    MATCHER_SCOPE(PagedAttentionFusion);
    using namespace ov::pass::pattern;

    auto q_hidden_size = ov::gen_pattern::Symbol("q_hidden_size");
    auto k_hidden_size = ov::gen_pattern::Symbol("k_hidden_size");
    auto v_hidden_size = ov::gen_pattern::Symbol("v_hidden_size");
    auto v_head_size = ov::gen_pattern::Symbol("v_head_size");
    auto out_hidden_size = ov::gen_pattern::Symbol("out_hidden_size");

    auto fc_output = any_input();
    auto aten_cat_Concat = makePattern<ov::op::internal::RoPE>({fc_output, any_input(), any_input()});
    auto Transpose_19027 = makePattern<opset1::Reshape>({aten_cat_Concat, {-1, 1, 0, 0}}, {{"special_zero", true}});
    auto Reshape_19029 = makePattern<opset1::Reshape>({Transpose_19027, {0, -1}}, {{"special_zero", true}});
    auto aten_cat_Concat_1 = makePattern<ov::op::internal::RoPE>({fc_output, any_input(), any_input()});
    auto Transpose_19032 = makePattern<opset1::Reshape>({aten_cat_Concat_1, {-1, 1, 0, 0}}, {{"special_zero", true}});
    auto Reshape_19039 = makePattern<opset1::Reshape>({Transpose_19032, {0, -1}}, {{"special_zero", true}});
    auto prim_ListUnpack = makePattern<opset1::VariadicSplit>({fc_output, -1, {q_hidden_size, k_hidden_size, v_hidden_size}});
    prim_ListUnpack->set_output_size(3);
    auto aten_view_Reshape_3 = makePattern<opset1::Reshape>({prim_ListUnpack->output(2),
        {0, 0, v_hidden_size / v_head_size, v_head_size}}, {{"special_zero", true}});
    auto Transpose_19036 = makePattern<opset1::Reshape>({aten_view_Reshape_3, {-1, 1, 0, 0}}, {{"special_zero", true}});
    auto Reshape_19041 = makePattern<opset1::Reshape>({Transpose_19036, {0,-1}}, {{"special_zero", true}});
    auto PagedAttentionExtension_19048 = makePattern<ov::op::PagedAttentionExtension>({Reshape_19029, Reshape_19039, Reshape_19041,
        any_input(), any_input(), any_input(), any_input(), any_input(), any_input(), any_input(), any_input(), any_input(), any_input()});
    auto Reshape_19059 = makePattern<opset1::Reshape>({PagedAttentionExtension_19048->output(0), {0, 1, -1, v_head_size}},
        {{"special_zero", true}});
    auto core_attention_aten_permute_Transpose_3 = makePattern<opset1::Reshape>({Reshape_19059, {1, -1, 0, 0}}, {{"special_zero", true}});
    auto core_attention_aten_reshape_Reshape = makePattern<opset1::Reshape>({core_attention_aten_permute_Transpose_3, {0, 0, out_hidden_size}},
        {{"special_zero", true}});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        auto old_node = as_type_ptr<ov::op::PagedAttentionExtension>(pattern_map.at(PagedAttentionExtension_19048).get_node_shared_ptr());
        if (!old_node)
            return false;
        auto args = old_node->input_values();
        args[0] = pattern_map.at(aten_cat_Concat);
        args[1] = pattern_map.at(aten_cat_Concat_1);
        args[2] = pattern_map.at(fc_output);
        auto pa_reshape_out = pattern_map.at(core_attention_aten_reshape_Reshape);
        Extensions::Cpu::PagedAttentionFuseConfig config;
        config.fuse_reshape_split = true;
        config.is_seq_len_first = true;
        config.slice_start = static_cast<size_t>(validator["q_hidden_size"] + validator["k_hidden_size"]);
        config.slice_stop = config.slice_start + static_cast<size_t>(validator["v_hidden_size"]);
        config.v_head_size = static_cast<size_t>(validator["v_head_size"]);
        config.out_hidden_size = static_cast<size_t>(validator["out_hidden_size"]);
        config.output_type[0] = pa_reshape_out.get_element_type();
        config.output_type[1] = old_node->get_output_element_type(1);
        auto new_node = std::make_shared<ov::intel_cpu::PagedAttentionWithSplit>(args, config);

        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info(old_node, new_node);
        ov::replace_node(old_node, new_node);

        auto present_to = pa_reshape_out.get_target_inputs();
        for (auto& to : present_to) {
            auto to_node = to.get_node();
            to_node->set_argument(to.get_index(), new_node->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(core_attention_aten_reshape_Reshape, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov
