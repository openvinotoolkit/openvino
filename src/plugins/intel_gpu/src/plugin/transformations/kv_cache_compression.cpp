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

    bool first = true;
    if (getenv("KV_CACHE_COMP") == nullptr) {
        if (first) {
            printf("NO_KV_CACHE_COMP\n");
        }
        first = false;
        return;
    }
    if (first) {
        printf("YES_KV_CACHE_COMP\n");
        first = false;
    }
    
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
        auto value_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(value).get_node_shared_ptr());
        auto org_sdpa = std::dynamic_pointer_cast<ov::intel_gpu::op::IndirectSDPA>(pattern_map.at(present).get_node_shared_ptr());
        
        if (true
            // || org_sdpa->get_friendly_name().find(".h.0.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.1.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.2.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.3.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.4.") != std::string::npos
        // || org_sdpa->get_friendly_name().find(".h.5.") != std::string::npos
            ) {
            std::cout << "pattern matched! " << org_sdpa->get_friendly_name() << std::endl;   
            auto k_dyn_quan = std::make_shared<op::DynamicQuantize>(key_node->get_input_node_shared_ptr(1));
            k_dyn_quan->set_friendly_name("dyn_quan_key");

            // FIXME: need to tell whether it is direct KV cache or indirect kv cache
            auto new_kv_cache_k = std::make_shared<op::KVCache>(key_node->get_input_node_shared_ptr(0),
                                                                k_dyn_quan->output(0),
                                                                key_node->get_input_node_shared_ptr(2),
                                                                k_dyn_quan->output(1),
                                                                key_node->get_variable(),
                                                                key_node->get_concat_axis(),
                                                                key_node->get_gather_axis(),
                                                                key_node->get_output_element_type(0));

            new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
            ov::copy_runtime_info(key_node, new_kv_cache_k);

            auto v_dyn_quan = std::make_shared<op::DynamicQuantize>(value_node->get_input_node_shared_ptr(1));
            v_dyn_quan->set_friendly_name("dyn_quan_value");
            // FIXME: need to tell whether it is direct KV cache or indirect kv cache
            auto new_kv_cache_v = std::make_shared<op::KVCache>(value_node->get_input_node_shared_ptr(0),
                                                                v_dyn_quan->output(0),
                                                                value_node->get_input_node_shared_ptr(2),
                                                                v_dyn_quan->output(1),
                                                                value_node->get_variable(),
                                                                value_node->get_concat_axis(),
                                                                value_node->get_gather_axis(),
                                                                value_node->get_output_element_type(0));

            new_kv_cache_v->set_friendly_name(value_node->get_friendly_name());
            ov::copy_runtime_info(value_node, new_kv_cache_v);

            // FIXME: output port from new_kv_cache_k is fixed. compression and indirectness is orthogonal.
            OutputVector sdpa_inputs;
            // QKV -- attention_mask -- input_scale -- key_scale -- beam_idx
            for (size_t i = 0; i < org_sdpa->get_input_size() - 1; i++)
                sdpa_inputs.push_back(org_sdpa->get_input_node_shared_ptr(i));
            sdpa_inputs[1] = new_kv_cache_k->output(0);         // compressed K
            sdpa_inputs[2] = new_kv_cache_v->output(0);         // compressed V
            sdpa_inputs.push_back(new_kv_cache_k->output(2));   // scale for compressed K
            sdpa_inputs.push_back(new_kv_cache_v->output(2));   // scale for compressed V

            // auto new_sdpa = org_sdpa->clone_with_new_inputs(sdpa_inputs);
            auto new_sdpa = std::make_shared<op::IndirectSDPA>(sdpa_inputs,
                                                               new_kv_cache_k->output(1),
                                                               org_sdpa->get_causal(),
                                                               true /* kv_compressed */,
                                                               org_sdpa->get_indirect_axis(),
                                                               org_sdpa->get_input0_transpose_order(),
                                                               org_sdpa->get_input1_transpose_order(),
                                                               org_sdpa->get_input2_transpose_order(),
                                                               org_sdpa->get_output_transpose_order(),
                                                               org_sdpa->get_output_type());

            new_kv_cache_k->set_friendly_name(key_node->get_friendly_name());
            ov::copy_runtime_info(key_node, new_kv_cache_k);

            new_kv_cache_v->set_friendly_name(value_node->get_friendly_name());
            ov::copy_runtime_info(value_node, new_kv_cache_v);

            new_sdpa->set_friendly_name(org_sdpa->get_friendly_name());
            ov::copy_runtime_info(org_sdpa, new_sdpa);

            ov::replace_node(org_sdpa, new_sdpa);
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
