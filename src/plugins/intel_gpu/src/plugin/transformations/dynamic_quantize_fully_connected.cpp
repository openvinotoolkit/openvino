// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_fully_connected.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

DynamicQuantizeFullyConnected::DynamicQuantizeFullyConnected(size_t group_size) {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto weights = any_input();
    auto fully_connected_compressed = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        // FIXME: need to handle group size
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        
        const auto& m_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(pattern_map.at(fully_connected_compressed).get_node_shared_ptr());
        std::cout << "pattern matched " << m_fc->get_friendly_name() << std::endl;
        OutputVector fc_inputs;
        auto dyn_quan = std::make_shared<op::DynamicQuantize>(m_data, group_size);
        for (size_t i = 0; i < m_fc->get_input_size(); i++)
            fc_inputs.push_back(m_fc->get_input_node_shared_ptr(i));
        fc_inputs[0] = dyn_quan->output(0);
        fc_inputs.push_back(dyn_quan->output(1));
        auto new_fc = std::make_shared<op::FullyConnectedCompressed>(fc_inputs,
                                                                     m_fc->get_has_zp(),
                                                                     m_fc->get_has_activation_scale(),
                                                                     m_fc->get_output_type());
        ov::replace_node(m_fc, new_fc);

        new_fc->set_friendly_name(m_fc->get_friendly_name());
        ov::copy_runtime_info(m_fc, new_fc);

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_compressed, "DynamicQuantizeFullyConnected");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
