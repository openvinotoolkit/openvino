// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"
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
#include "openvino/op/transpose.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"

namespace ov {
namespace intel_gpu {

class KVCacheCompressionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KVCacheCompressionMatcher", "0");
    KVCacheCompressionMatcher();
};

KVCacheCompressionMatcher::KVCacheCompressionMatcher() {
    using namespace ov::pass::pattern;
    
    auto query = any_input();

    auto k_past = any_input();
    auto k_new_token = any_input();
    auto k_beam_idx = any_input();
    auto key = wrap_type<ov::intel_gpu::op::KVCache>({k_past, k_new_token, k_beam_idx});

    auto v_past = any_input();
    auto v_new_token = any_input();
    auto v_beam_idx = any_input();
    auto value = wrap_type<ov::intel_gpu::op::KVCache>({v_past, v_new_token, v_beam_idx});

    auto input_attn_mask = any_input();
    auto input_scale = any_input();

    auto present = wrap_type<ov::intel_gpu::op::IndirectSDPA>({query, key, value, input_attn_mask, input_scale});

    // k, v, attention_mask, scale
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        
        auto k_new_token_node = pattern_map.at(k_new_token).get_node_shared_ptr();
        auto key_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(key).get_node_shared_ptr());
        auto present_node = std::dynamic_pointer_cast<ov::intel_gpu::op::IndirectSDPA>(pattern_map.at(present).get_node_shared_ptr());
        
        if (present_node->get_friendly_name().find("__module.model.transformer.h.0.attn/aten::scaled_dot_product_attention/ScaledDotProductAttention") != std::string::npos) {
            std::cout << "pattern matched! " << key_node->get_friendly_name() << std::endl;   
            auto new_dyn_quan = std::make_shared<op::DynamicQuantize>(key_node->get_input_node_shared_ptr(1));
            // FIXME: need to tell whether it is direct KV cache or indirect kv cache
            auto new_kv_cache_k = std::make_shared<op::KVCache>(key_node->get_input_node_shared_ptr(0),
                                                                new_dyn_quan->output(0),
                                                                key_node->get_input_node_shared_ptr(2),
                                                                new_dyn_quan->output(1),
                                                                key_node->get_variable(),
                                                                key_node->get_concat_axis(),
                                                                key_node->get_gather_axis(),
                                                                key_node->get_output_element_type(0));
            new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
            ov::copy_runtime_info(key_node, new_kv_cache_k);
            ov::replace_node(key_node, new_kv_cache_k);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(present, "KVCacheCompressionMatcher");
    this->register_matcher(m, callback);

}

bool KVCacheCompression::run_on_model(const std::shared_ptr<ov::Model>& m) {
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

KVCacheCompression::KVCacheCompression() {
    add_matcher<ov::intel_gpu::KVCacheCompressionMatcher>();
}

}  // namespace intel_gpu
}  // namespace ov
