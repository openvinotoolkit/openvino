// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_convert_fusion.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

FullyConnectedConvertFusion::FullyConnectedConvertFusion() {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto weights = any_input();
    auto bias = any_input();
    auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias}, consumers_count(1));
    auto fully_connected_compressed1 = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input()}, consumers_count(1));
    auto fully_connected_compressed2 = wrap_type<op::FullyConnectedCompressed>({data, weights, bias, any_input(), any_input()}, consumers_count(1));
    auto fully_connected_compressed = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected_compressed1, fully_connected_compressed2});
    auto fc = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected, fully_connected_compressed});
    auto convert = wrap_type<ov::op::v0::Convert>({fc}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_bias = pattern_map.at(bias).get_node_shared_ptr();
        const auto& m_convert = pattern_map.at(convert).get_node_shared_ptr();
        auto output_type = m_convert->get_output_element_type(0);

        std::shared_ptr<Node> m_fc = nullptr;
        std::shared_ptr<Node> new_fc = nullptr;
        auto it = pattern_map.find(fully_connected);
        if (it != pattern_map.end()) {
            m_fc = it->second.get_node_shared_ptr();
            new_fc = std::make_shared<op::FullyConnected>(m_data, m_weights, m_bias, output_type);
        } else {
            m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
            if (m_fc->input_values().size() == 4)
                new_fc = std::make_shared<op::FullyConnectedCompressed>(m_data,
                                                                        m_weights,
                                                                        m_bias,
                                                                        m_fc->input_value(3),
                                                                        output_type);
            else
                new_fc = std::make_shared<op::FullyConnectedCompressed>(m_data,
                                                                        m_weights,
                                                                        m_bias,
                                                                        m_fc->input_value(3),
                                                                        m_fc->input_value(4),
                                                                        output_type);
        }
        new_fc->set_friendly_name(m_convert->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), new_fc);
        replace_node(m_convert, new_fc);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, "FullyConnectedConvertFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
