// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/pa_kv_reorder.hpp"

#include <algorithm>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"

namespace ov {
namespace op {
namespace internal {
using PA_KV_Reorder = ov::intel_gpu::op::PA_KV_Reorder;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreatePA_KV_ReorderOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PA_KV_Reorder>& op) {
    validate_inputs_count(op, {6});

    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::pa_kv_reorder(cldnn::primitive_id(layer_type_name_ID(op)), std::vector<cldnn::input_info>(inputs.begin(), inputs.end()));
    const auto& rt_info = op->get_rt_info();
    const auto k_head_size_id = "k_head_size";
    const auto v_head_size_id = "v_head_size";
    const auto num_k_heads_id = "num_k_heads";
    const auto has_rt_head_info =
        rt_info.find(k_head_size_id) != rt_info.end() && rt_info.find(v_head_size_id) != rt_info.end() && rt_info.find(num_k_heads_id) != rt_info.end();
    if (!has_rt_head_info)
        OPENVINO_THROW("[GPU] pa_kv_reorder op missing runtime head size information in rt_info");

    constexpr const char* rt_is_key_by_channel = "pa_kv_reorder.is_key_by_channel";
    constexpr const char* rt_scales_zp_size = "pa_kv_reorder.scales_zp_size";
    constexpr const char* rt_key_dim_order = "pa_kv_reorder.key_cache_dim_order";
    constexpr const char* rt_value_dim_order = "pa_kv_reorder.value_cache_dim_order";
    const auto has_rt_compress_info = rt_info.find(rt_is_key_by_channel) != rt_info.end() && rt_info.find(rt_scales_zp_size) != rt_info.end() &&
                                      rt_info.find(rt_key_dim_order) != rt_info.end() && rt_info.find(rt_value_dim_order) != rt_info.end();

    if (!has_rt_compress_info) {
        OPENVINO_THROW("[GPU] pa_kv_reorder op missing runtime compression information in rt_info");
    }
    prim.kv_heads_num = static_cast<size_t>(rt_info.at(num_k_heads_id).as<int64_t>());
    prim.adjusted_paged_attention_block_size = cldnn::paged_attention::block_size;
    prim.adjusted_k_head_size = static_cast<size_t>(rt_info.at(k_head_size_id).as<int64_t>());
    prim.adjusted_v_head_size = static_cast<size_t>(rt_info.at(v_head_size_id).as<int64_t>());

    prim.cache_dt = cldnn::element_type_to_data_type(op->get_input_element_type(cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE));
    prim.is_kv_compressed = prim.cache_dt == cldnn::data_types::i8 || prim.cache_dt == cldnn::data_types::u8;

    if (prim.is_kv_compressed) {
        prim.scales_zp_size = static_cast<size_t>(rt_info.at(rt_scales_zp_size).as<int64_t>());
        if (rt_info.at(rt_is_key_by_channel).as<bool>()) {
            prim.is_key_by_channel = true;
            prim.adjusted_paged_attention_block_size = prim.adjusted_paged_attention_block_size + prim.scales_zp_size;
        } else {
            prim.adjusted_k_head_size = prim.adjusted_k_head_size + prim.scales_zp_size;
        }
        prim.adjusted_v_head_size = prim.adjusted_v_head_size + prim.scales_zp_size;
        const auto key_dim_order = rt_info.at(rt_key_dim_order).as<std::vector<size_t>>();
        const auto value_dim_order = rt_info.at(rt_value_dim_order).as<std::vector<size_t>>();
        OPENVINO_ASSERT(key_dim_order.size() == 4 && value_dim_order.size() == 4, "[GPU] Invalid pa_kv_reorder dim order metadata size");

        std::copy(key_dim_order.begin(), key_dim_order.end(), prim.key_cache_dim_order.begin());
        std::copy(value_dim_order.begin(), value_dim_order.end(), prim.value_cache_dim_order.begin());
    }

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PA_KV_Reorder);

}  // namespace ov::intel_gpu
