// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "indirect_kv_cache.hpp"
#include <memory>

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/indirect_gemm.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace {
// same impl as ov::replace node, but w/o outputs count check
void replace_node_unsafe(const std::shared_ptr<ov::Node>& target, const std::shared_ptr<ov::Node>& replacement) {
    if (ov::op::util::is_output(target)) {
        OPENVINO_THROW("Result nodes cannot be replaced.");
    }
    for (size_t i = 0; i < target->get_output_size(); i++) {
        target->output(i).replace(replacement->output(0));
    }

    replacement->add_node_control_dependents(target);
    replacement->add_node_control_dependencies(target);
    target->clear_control_dependents();
}

}  // namespace

namespace ov {
namespace intel_gpu {

IndirectKVCache::IndirectKVCache() {
    using namespace ov::pass::pattern;

    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_input = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto gather_past = wrap_type<ov::op::v8::Gather>({gather_input, beam_idx, wrap_type<ov::op::v0::Constant>()});
    auto kv_cache = wrap_type<ov::intel_gpu::op::KVCache>({gather_past, any_input()});
    auto matmul_0 = wrap_type<ov::intel_gpu::op::Gemm>({kv_cache, any_input()});
    auto matmul_1 = wrap_type<ov::intel_gpu::op::Gemm>({any_input(), kv_cache});
    auto matmul = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{matmul_0, matmul_1});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto kv_cache_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(kv_cache).get_node_shared_ptr());

        auto beam_idx_node = pattern_map.at(beam_idx).get_node_shared_ptr();
        auto gather_input_node = pattern_map.at(gather_input).get_node_shared_ptr();
        auto gather_node = pattern_map.at(gather_past).get_node_shared_ptr();
        ov::replace_node(gather_node, gather_input_node);

        auto indirect_kv_cache = std::make_shared<op::KVCache>(gather_input_node,
                                                               kv_cache_node->get_input_node_shared_ptr(1),
                                                               beam_idx_node,
                                                               kv_cache_node->get_variable(),
                                                               kv_cache_node->get_concat_axis(),
                                                               kv_cache_node->get_gather_axis(),
                                                               kv_cache_node->get_output_element_type(0));

        indirect_kv_cache->set_friendly_name(kv_cache_node->get_friendly_name());
        ov::copy_runtime_info(kv_cache_node, indirect_kv_cache);
        replace_node_unsafe(kv_cache_node, indirect_kv_cache);

        auto kv_cache_users = indirect_kv_cache->get_output_target_inputs(0);
        auto matmul_kv_cache_index = kv_cache_users.begin()->get_index();

        auto gemm_node = std::dynamic_pointer_cast<op::Gemm>(m.get_match_root());
        auto order_in0 = gemm_node->get_input0_order();
        auto order_in1 = gemm_node->get_input1_order();
        auto order_out = gemm_node->get_output_order();

        auto indirect_gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(gemm_node->get_input_node_shared_ptr(0),
                                                                               gemm_node->get_input_node_shared_ptr(1),
                                                                               indirect_kv_cache->output(1), // beam table
                                                                               matmul_kv_cache_index == 0,
                                                                               matmul_kv_cache_index == 1,
                                                                               order_in0,
                                                                               order_in1,
                                                                               order_out);

        indirect_gemm->set_friendly_name(gemm_node->get_friendly_name());
        ov::copy_runtime_info(gemm_node, indirect_gemm);
        ov::replace_node(gemm_node, indirect_gemm);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul, "IndirectKVCache");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
