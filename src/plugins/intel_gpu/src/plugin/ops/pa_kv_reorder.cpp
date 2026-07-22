// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pa_kv_reorder.hpp"

#include <algorithm>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov {
namespace op {
namespace internal {
using PA_KV_Reorder = ov::op::internal::PaKVReorder;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

namespace {

bool is_compressed_cache_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 || type == ov::element::i4 || type == ov::element::u4;
}

size_t infer_scales_zp_size(const ov::element::Type& cache_type, const ov::element::Type& infer_precision) {
    if (!is_compressed_cache_type(cache_type)) {
        return 0;
    }
    // One scale + one zp (each `infer_precision`-typed) per quantization group, matching
    // pa_kv_cache_update kernel which writes comp_ptr[token]=1/scale, comp_ptr[BLOCK+token]=zp.
    if (cache_type == ov::element::i4 || cache_type == ov::element::u4) {
        return 2 * infer_precision.size();
    }
    return (2 * infer_precision.size()) / cache_type.size();
}

}  // namespace

static void CreatePA_KV_ReorderOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PA_KV_Reorder>& op) {
    validate_inputs_count(op, {6});

    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::pa_kv_reorder(cldnn::primitive_id(layer_type_name_ID(op)), std::vector<cldnn::input_info>(inputs.begin(), inputs.end()));

    const auto& config = p.get_config();
    // Use configured kv_cache_precision (e.g. u4) — NOT the parameter's element type, which
    // ConvertPagedAttnInputs rewrites to i8/u8 for RemoteTensor compatibility even when the
    // underlying cache layout is packed u4.
    const auto cache_type = config.get_kv_cache_precision();
    const auto infer_precision = config.get_inference_precision();
    const auto key_cache_quant_mode = config.get_key_cache_quant_mode();

    prim.cache_dt = cldnn::element_type_to_data_type(cache_type);
    prim.scales_zp_size = infer_scales_zp_size(cache_type, infer_precision);
    prim.is_kv_compressed = prim.scales_zp_size > 0;
    prim.is_key_by_channel = (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL);

    const auto key_cache_ps = op->get_input_partial_shape(cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE);
    const auto value_cache_ps = op->get_input_partial_shape(cldnn::pa_kv_reorder::PaKVReorderInputIdx::VALUE_CACHE);

    OPENVINO_ASSERT(key_cache_ps.rank().is_static() && key_cache_ps.rank().get_length() == 4,
                    "[GPU] pa_kv_reorder expects 4D key cache, got rank ", key_cache_ps.rank());
    OPENVINO_ASSERT(value_cache_ps.rank().is_static() && value_cache_ps.rank().get_length() == 4,
                    "[GPU] pa_kv_reorder expects 4D value cache, got rank ", value_cache_ps.rank());

    const auto& rt_info = op->get_rt_info();
    const auto k_head_size_id = "k_head_size";
    const auto v_head_size_id = "v_head_size";
    const auto num_k_heads_id = "num_k_heads";

    OPENVINO_ASSERT(rt_info.find(k_head_size_id) != rt_info.end() &&
                    rt_info.find(v_head_size_id) != rt_info.end() &&
                    rt_info.find(num_k_heads_id) != rt_info.end(),
                    "[GPU] pa_kv_reorder: input shapes are dynamic but rt_info is missing "
                    "k_head_size/v_head_size/num_k_heads");

    const size_t k_head_size = rt_info.at(k_head_size_id).as<int64_t>();
    const size_t v_head_size = rt_info.at(v_head_size_id).as<int64_t>();
    const size_t kv_heads_num = rt_info.at(num_k_heads_id).as<int64_t>();
    const size_t block_size = cldnn::paged_attention::block_size;

    prim.kv_heads_num = kv_heads_num;
    if (prim.is_kv_compressed) {
        if (prim.is_key_by_channel) {
            prim.adjusted_k_head_size = k_head_size;
            prim.adjusted_paged_attention_block_size = block_size + prim.scales_zp_size;
        } else {
            prim.adjusted_k_head_size = k_head_size + prim.scales_zp_size;
            prim.adjusted_paged_attention_block_size = block_size;
        }
        prim.adjusted_v_head_size = v_head_size + prim.scales_zp_size;
    } else {
        prim.adjusted_k_head_size = k_head_size;
        prim.adjusted_paged_attention_block_size = block_size;
        prim.adjusted_v_head_size = v_head_size;
    }

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PA_KV_Reorder);

}  // namespace ov::intel_gpu
