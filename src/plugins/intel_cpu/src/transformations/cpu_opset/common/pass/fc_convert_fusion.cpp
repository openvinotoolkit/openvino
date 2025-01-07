// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_convert_fusion.hpp"

#include <utils/general_utils.h>

#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace intel_cpu {

FcConvertFusion::FcConvertFusion() {
    MATCHER_SCOPE(FcConvertFusion);
    using namespace ov::pass::pattern;

    auto a = any_input();
    auto b = any_input();
    auto fc = wrap_type<ov::op::internal::FullyConnected>({a, b}, consumers_count(1));
    auto convert = wrap_type<ov::op::v0::Convert>({fc}, type_matches(ov::element::f32));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& m_a = pattern_map.at(a).get_node_shared_ptr();
        const auto& m_b = pattern_map.at(b).get_node_shared_ptr();
        const auto& m_fc = pattern_map.at(fc).get_node_shared_ptr();

        if (!one_of(m_a->get_output_element_type(0), ov::element::f16, ov::element::bf16, ov::element::f32) &&
            !one_of(m_b->get_output_element_type(0), ov::element::f16, ov::element::bf16, ov::element::f32)) {
            return false;
        }

        const auto out = m_fc->outputs();
        const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
        if (!has_only_child) {
            return false;
        }

        const auto& m_convert = pattern_map.at(convert).get_node_shared_ptr();
        auto output_type = m_convert->get_output_element_type(0);
        auto new_fc = std::make_shared<ov::op::internal::FullyConnected>(m_a, m_b, output_type);

        new_fc->set_friendly_name(m_convert->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), new_fc);
        replace_node(m_convert, new_fc);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov