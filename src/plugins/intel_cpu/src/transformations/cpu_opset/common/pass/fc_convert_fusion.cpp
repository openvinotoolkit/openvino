// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_convert_fusion.hpp"

#include <utils/general_utils.h>

#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

namespace ov {
namespace intel_cpu {

FullyConnectedConvertFusion::FullyConnectedConvertFusion() {
    MATCHER_SCOPE(FullyConnectedConvertFusion);
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto weights = any_input();
    auto fully_connected1 = wrap_type<ov::op::internal::FullyConnected>({data, weights}, consumers_count(1));
    auto fully_connected2 = wrap_type<ov::op::internal::FullyConnected>({data, weights, any_input()}, consumers_count(1));
    auto fully_connected = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected1, fully_connected2});
    auto fully_connected_compressed1 = wrap_type<ov::op::internal::FullyConnectedCompressed>({data, weights, any_input(), any_input()}, consumers_count(1));
    auto fully_connected_compressed2 = wrap_type<ov::op::internal::FullyConnectedCompressed>({data, weights, any_input(), any_input(), any_input()}, consumers_count(1));
    auto fully_connected_compressed = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected_compressed1, fully_connected_compressed2});
    auto fc = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected, fully_connected_compressed});
    auto convert = wrap_type<ov::op::v0::Convert>({fc}, type_matches(ov::element::f32));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_convert = pattern_map.at(convert).get_node_shared_ptr();
        auto output_type = m_convert->get_output_element_type(0);

        if (!one_of(m_data->get_output_element_type(0), ov::element::f16, ov::element::bf16, ov::element::f32) &&
            !one_of(m_weights->get_output_element_type(0), ov::element::f16, ov::element::bf16, ov::element::f32)) {
            return false;
        }

        std::shared_ptr<Node> m_fc = nullptr;
        std::shared_ptr<Node> new_fc = nullptr;

        auto it = pattern_map.find(fully_connected);
        if (it != pattern_map.end()) {
            m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();
            if (m_fc->input_values().size() == 2) {
                new_fc = std::make_shared<ov::op::internal::FullyConnected>(m_data, m_weights, output_type);
            } else {
                new_fc = std::make_shared<ov::op::internal::FullyConnected>(m_data, m_weights, m_fc->input_value(2), output_type);
            }
        } else {
            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
            if (m_fc->input_values().size() == 4) {
                new_fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(m_data,
                                                                                      m_weights,
                                                                                      m_fc->input_value(2),
                                                                                      m_fc->input_value(3),
                                                                                      output_type);
            } else {
                new_fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(m_data,
                                                                                      m_weights,
                                                                                      m_fc->input_value(2),
                                                                                      m_fc->input_value(3),
                                                                                      m_fc->input_value(4),
                                                                                      output_type);
            }
        }

        const auto out = m_fc->outputs();
        const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
        if (!has_only_child) {
            return false;
        }

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