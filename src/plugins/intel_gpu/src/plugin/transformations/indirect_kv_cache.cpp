// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "indirect_kv_cache.hpp"
#include <memory>

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/indirect_gemm.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/op_types.hpp"
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

IndirectGemmOpt::IndirectGemmOpt() {
    using namespace ov::pass::pattern;

    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_input = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto axis_const = wrap_type<ov::op::v0::Constant>(
        ov::op::util::constant_predicate<int64_t>([](const std::vector<int64_t>& value) -> bool {
            return value.size() == 1 && (value[0] == 0 || value[0] == 1);
        }));
    auto gather_past = wrap_type<ov::op::v8::Gather>({gather_input, beam_idx, axis_const});
    auto kv_cache = wrap_type<ov::intel_gpu::op::KVCache>({gather_past, any_input()});
    auto matmul_0 = wrap_type<ov::intel_gpu::op::Gemm>({kv_cache, any_input()});
    auto matmul_1 = wrap_type<ov::intel_gpu::op::Gemm>({any_input(), kv_cache});
    auto matmul = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{matmul_0, matmul_1});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto kv_cache_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(kv_cache).get_node_shared_ptr());

        auto beam_idx_node = pattern_map.at(beam_idx).get_node_shared_ptr();
        auto gather_input_node = pattern_map.at(gather_input).get_node_shared_ptr();
        auto gather_node = std::dynamic_pointer_cast<ov::op::v8::Gather>(pattern_map.at(gather_past).get_node_shared_ptr());
        auto gather_axis = gather_node->get_axis();
        ov::replace_node(gather_node, gather_input_node);

        auto indirect_kv_cache = std::make_shared<op::KVCache>(gather_input_node,
                                                               kv_cache_node->input(1).get_source_output(),
                                                               beam_idx_node,
                                                               kv_cache_node->get_variable(),
                                                               kv_cache_node->get_concat_axis(),
                                                               gather_axis,
                                                               kv_cache_node->get_output_element_type(0));

        indirect_kv_cache->set_friendly_name(kv_cache_node->get_friendly_name());
        ov::copy_runtime_info(kv_cache_node, indirect_kv_cache);
        replace_node_unsafe(kv_cache_node, indirect_kv_cache);

        auto kv_cache_users = indirect_kv_cache->get_output_target_inputs(0);
        auto matmul_kv_cache_index = kv_cache_users.begin()->get_index();

        auto gemm_node = std::dynamic_pointer_cast<op::Gemm>(m.get_match_root());
        auto order_in0 = gemm_node->get_input0_transpose_order();
        auto order_in1 = gemm_node->get_input1_transpose_order();
        auto order_out = gemm_node->get_output_transpose_order();

        auto indirect_gemm = std::make_shared<ov::intel_gpu::op::IndirectGemm>(gemm_node->get_input_node_shared_ptr(0),
                                                                               gemm_node->get_input_node_shared_ptr(1),
                                                                               indirect_kv_cache->output(1), // beam table
                                                                               matmul_kv_cache_index == 0,
                                                                               matmul_kv_cache_index == 1,
                                                                               gather_axis,
                                                                               order_in0,
                                                                               order_in1,
                                                                               order_out);

        indirect_gemm->set_friendly_name(gemm_node->get_friendly_name());
        ov::copy_runtime_info(gemm_node, indirect_gemm);
        ov::replace_node(gemm_node, indirect_gemm);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul, "IndirectGemmOpt");
    this->register_matcher(m, callback);
}

IndirectSDPAOpt::IndirectSDPAOpt() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_input_0 = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto gather_input_1 = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto axis_const = wrap_type<ov::op::v0::Constant>(
        ov::op::util::constant_predicate<int64_t>([](const std::vector<int64_t>& value) -> bool {
            return value.size() == 1 && (value[0] == 0 || value[0] == 1);
        }));
    auto gather_past_0 = wrap_type<ov::op::v8::Gather>({gather_input_0, beam_idx, axis_const});
    auto gather_past_1 = wrap_type<ov::op::v8::Gather>({gather_input_1, beam_idx, axis_const});
    auto kv_cache_0 = wrap_type<ov::intel_gpu::op::KVCache>({gather_past_0, any_input()});
    auto kv_cache_1 = wrap_type<ov::intel_gpu::op::KVCache>({gather_past_1, any_input()});

    auto input_attn_mask = any_input();
    auto input_scale = any_input();
    auto sdpa_without_attn_mask_m = wrap_type<ov::intel_gpu::op::SDPA>({ any_input(), kv_cache_0, kv_cache_1 });
    auto sdpa_with_attn_mask_m = wrap_type<ov::intel_gpu::op::SDPA>({ any_input(), kv_cache_0, kv_cache_1, input_attn_mask });
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<ov::intel_gpu::op::SDPA>({ any_input(), kv_cache_0, kv_cache_1, input_attn_mask, input_scale });

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto kv_cache_node_0 = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(kv_cache_0).get_node_shared_ptr());
        auto kv_cache_node_1 = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(kv_cache_1).get_node_shared_ptr());

        auto beam_idx_node = pattern_map.at(beam_idx).get_node_shared_ptr();
        auto gather_input_node_0 = pattern_map.at(gather_input_0).get_node_shared_ptr();
        auto gather_input_node_1 = pattern_map.at(gather_input_1).get_node_shared_ptr();
        auto gather_node_0 = std::dynamic_pointer_cast<ov::op::v8::Gather>(pattern_map.at(gather_past_0).get_node_shared_ptr());
        auto gather_node_1 = std::dynamic_pointer_cast<ov::op::v8::Gather>(pattern_map.at(gather_past_1).get_node_shared_ptr());
        auto gather_axis_0 = gather_node_0->get_axis();
        auto gather_axis_1 = gather_node_1->get_axis();
        OPENVINO_ASSERT(gather_axis_0 == gather_axis_1);

        ov::replace_node(gather_node_0, gather_input_node_0);
        ov::replace_node(gather_node_1, gather_input_node_1);

        auto indirect_kv_cache_0 = std::make_shared<op::KVCache>(gather_input_node_0,
                                                                 kv_cache_node_0->get_input_node_shared_ptr(1),
                                                                 beam_idx_node,
                                                                 kv_cache_node_0->get_variable(),
                                                                 kv_cache_node_0->get_concat_axis(),
                                                                 gather_axis_0,
                                                                 kv_cache_node_0->get_output_element_type(0));

        auto indirect_kv_cache_1 = std::make_shared<op::KVCache>(gather_input_node_1,
                                                                 kv_cache_node_1->get_input_node_shared_ptr(1),
                                                                 beam_idx_node,
                                                                 kv_cache_node_1->get_variable(),
                                                                 kv_cache_node_1->get_concat_axis(),
                                                                 gather_axis_1,
                                                                 kv_cache_node_1->get_output_element_type(0));

        indirect_kv_cache_0->set_friendly_name(kv_cache_node_0->get_friendly_name());
        indirect_kv_cache_1->set_friendly_name(kv_cache_node_1->get_friendly_name());
        ov::copy_runtime_info(kv_cache_node_0, indirect_kv_cache_0);
        ov::copy_runtime_info(kv_cache_node_1, indirect_kv_cache_1);
        replace_node_unsafe(kv_cache_node_0, indirect_kv_cache_0);
        replace_node_unsafe(kv_cache_node_1, indirect_kv_cache_1);

        auto sdpa = std::dynamic_pointer_cast<op::SDPA>(m.get_match_root());
        auto order_in0 = sdpa->get_input0_transpose_order();
        auto order_in1 = sdpa->get_input1_transpose_order();
        auto order_in2 = sdpa->get_input2_transpose_order();
        auto order_out = sdpa->get_output_transpose_order();
        auto is_causal = sdpa->get_causal();

        OutputVector data_inputs;
        data_inputs.push_back(sdpa->get_input_node_shared_ptr(0)); // Q
        data_inputs.push_back(sdpa->get_input_node_shared_ptr(1)); // K
        data_inputs.push_back(sdpa->get_input_node_shared_ptr(2)); // V

        if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            data_inputs.push_back(sdpa->get_input_source_output(3));
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            data_inputs.push_back(sdpa->get_input_source_output(3));
            data_inputs.push_back(sdpa->get_input_source_output(4));
        }

        auto indirect_sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(data_inputs,
                                                                               indirect_kv_cache_0->output(1), // beam table
                                                                               is_causal,
                                                                               gather_axis_1,
                                                                               order_in0,
                                                                               order_in1,
                                                                               order_in2,
                                                                               order_out);

        OPENVINO_ASSERT(indirect_sdpa != nullptr);

        indirect_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(sdpa, indirect_sdpa);
        ov::replace_node(sdpa, indirect_sdpa);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "IndirectSDPAOpt");
    this->register_matcher(m, callback);
}

IndirectKVCache::IndirectKVCache() {
    add_matcher<IndirectGemmOpt>();
    add_matcher<IndirectSDPAOpt>();
}
}  // namespace intel_gpu
}  // namespace ov
