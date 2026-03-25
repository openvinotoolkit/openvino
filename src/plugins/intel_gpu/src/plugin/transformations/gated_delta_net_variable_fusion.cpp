// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net_variable_fusion.hpp"

#include <memory>

#include "intel_gpu/op/read_value.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

GatedDeltaNetVariableFusionMatcher::GatedDeltaNetVariableFusionMatcher() {
    using namespace ov::pass::pattern;

    auto query = any_input();
    auto key = any_input();
    auto value = any_input();
    auto gate = any_input();
    auto beta = any_input();

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto gather = wrap_type<ov::op::v8::Gather>({past, any_input(), 0}, {{"batch_dims", 0}});

    auto gdn = wrap_type<ov::op::internal::GatedDeltaNet>({query, key, value, gather, gate, beta});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {

        const auto& pattern_map = m.get_pattern_value_map();

        auto past_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past).get_node_shared_ptr());
        // auto assign_node = ov::as_type_ptr<ov::op::v6::Assign>(pattern_map.at(assign).get_node_shared_ptr());
        auto old_gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(pattern_map.at(gdn).get_node_shared_ptr());
        std::cout << "Start|Fused GatedDeltaNet with variable: " << old_gdn->get_friendly_name() << std::endl;

        auto variable = past_node->get_variable();

        auto new_gdn = std::make_shared<ov::op::internal::GatedDeltaNetWithVariable>(old_gdn->input_values(),
                                                                                      variable,
                                                                                      old_gdn->get_fuse_qk_l2norm(),
                                                                                      old_gdn->get_q_l2_norm_eps(),
                                                                                      old_gdn->get_k_l2_norm_eps());
        new_gdn->set_friendly_name(old_gdn->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), new_gdn);
        ov::replace_node(old_gdn, new_gdn);
        std::cout << "Fused GatedDeltaNet with variable: " << std::endl;
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gdn, "GatedDeltaNetVariableFusionMatcher");
    this->register_matcher(matcher, callback);
}

bool GatedDeltaNetVariableFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool changed = pass::GraphRewrite::run_on_model(m);
    std::cout << "GatedDeltaNetVariableFusion: " << (changed ? "changed" : "not changed") << std::endl;

    if (changed) {
        ov::SinkVector sinks = m->get_sinks();
        for (auto& sink : sinks) {
            auto assign = ov::as_type_ptr<ov::op::v6::Assign>(sink);
            if (!assign) {
                continue;
            }

            auto input_node = assign->input_value(0).get_node_shared_ptr();
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(input_node)) {
                input_node = convert->input_value(0).get_node_shared_ptr();
            }

            if (ov::is_type<ov::op::internal::GatedDeltaNetWithVariable>(input_node)) {
                m->remove_sink(sink);
            }
        }
    }

    return changed;
}

GatedDeltaNetVariableFusion::GatedDeltaNetVariableFusion() {
    add_matcher<ov::intel_gpu::GatedDeltaNetVariableFusionMatcher>();
}

}  // namespace ov::intel_gpu
