// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_fusion.hpp"
#include <memory>

#include "intel_gpu/op/kv_cache.hpp"

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

class KVCacheFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KVCacheFusionMatcher", "0");
    KVCacheFusionMatcher();
};

KVCacheFusionMatcher::KVCacheFusionMatcher() {
    using namespace ov::pass::pattern;

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto convert_past = wrap_type<ov::op::v0::Convert>({past});
    auto gather_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, convert_past});
    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_past = wrap_type<ov::op::v8::Gather>({gather_input, beam_idx, wrap_type<ov::op::v0::Constant>()});
    auto concat_past_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, convert_past, gather_past});
    auto concat = wrap_type<ov::op::v0::Concat>({concat_past_input, any_input()});
    auto convert_present = wrap_type<ov::op::v0::Convert>({concat});
    auto present_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{concat, convert_present});
    auto present = wrap_type<ov::op::v6::Assign>({present_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = std::dynamic_pointer_cast<ov::op::v0::Concat>(pattern_map.at(concat).get_node_shared_ptr());

        auto past_node = std::dynamic_pointer_cast<ov::op::v6::ReadValue>(pattern_map.at(past).get_node_shared_ptr());
        auto present_node = std::dynamic_pointer_cast<ov::op::v6::Assign>(pattern_map.at(present).get_node_shared_ptr());

        if (past_node->get_variable_id() != present_node->get_variable_id())
            return false;

        // TODO: Support conversion internally
        if (concat_node->get_output_element_type(0) != past_node->get_output_element_type(0))
            return false;

        auto variable = past_node->get_variable();
        auto concat_axis = concat_node->get_axis();

        std::shared_ptr<ov::Node> variable_initializer = nullptr;
        std::shared_ptr<ov::Node> kv_cache_node = nullptr;
        if (past_node->get_input_size() == 1) {
            variable_initializer = past_node->get_input_node_shared_ptr(0);
        }

        // Replace common ReadValue op with a custom one as common one expects paired Assign operation which is removed by this transform
        auto new_read_value_node = variable_initializer ? std::make_shared<ov::intel_gpu::op::ReadValue>(variable_initializer, variable)
                                                        : std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        new_read_value_node->set_friendly_name(past_node->get_friendly_name());
        ov::copy_runtime_info(past_node, new_read_value_node);
        ov::replace_node(past_node, new_read_value_node);

        if (pattern_map.count(gather_past) > 0) {
            kv_cache_node = std::make_shared<op::KVCache>(pattern_map.at(gather_past).get_node_shared_ptr(),
                                                          concat_node->input(1).get_source_output(),
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        } else {
            kv_cache_node = std::make_shared<op::KVCache>(new_read_value_node,
                                                          concat_node->input(1).get_source_output(),
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        }
        kv_cache_node->set_friendly_name(concat_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), kv_cache_node);
        ov::replace_node(concat_node, kv_cache_node);

        if (pattern_map.count(convert_present) > 0) {
            present_node->set_argument(0, kv_cache_node->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(present, "KVCacheFusionMatcher");
    this->register_matcher(m, callback);
}

bool KVCacheFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool res = pass::GraphRewrite::run_on_model(m);
    if (res) {
        ov::SinkVector sinks = m->get_sinks();
        for (auto& sink : sinks) {
            if (sink && sink->get_input_node_ptr(0)->get_type_info() == op::KVCache::get_type_info_static()) {
                m->remove_sink(sink);
            }
        }
    }

    return res;
}

KVCacheFusion::KVCacheFusion() {
    add_matcher<ov::intel_gpu::KVCacheFusionMatcher>();
}

}  // namespace intel_gpu
}  // namespace ov
