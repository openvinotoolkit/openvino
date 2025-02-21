// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_fuse_convert.hpp"

#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"

using namespace ov;
using namespace ov::pass::pattern;

intel_cpu::MLPFuseConvert::MLPFuseConvert() {
    MATCHER_SCOPE(MLPFuseConvert);

    auto mlp = wrap_type<ov::intel_cpu::LLMMLPNode>();
    auto convert = wrap_type<ov::op::v0::Convert>({mlp}, type_matches(ov::element::f32));

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        const auto& m_mlp = pattern_map.at(mlp).get_node_shared_ptr();
        const auto& m_cvt = pattern_map.at(convert).get_node_shared_ptr();

        auto mlp_node = as_type_ptr<ov::intel_cpu::LLMMLPNode>(m_mlp);
        if (!mlp_node) {
            return false;
        }
        const auto out = mlp_node->outputs();
        const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
        if (!has_only_child) {
            return false;
        }

        OutputVector args = mlp_node->get_args();
        const auto cfg = mlp_node->get_config();

        auto new_mlp = std::make_shared<ov::intel_cpu::LLMMLPNode>(args, cfg, ov::element::f32);

        copy_runtime_info(m_cvt, new_mlp);
        ov::replace_node(m_cvt, new_mlp);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}