// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sdpa_base.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "sdpa_utils.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

static std::string get_broadcast_input_str(const size_t input_rank, const int64_t axes, const int64_t val) {
    std::vector<std::string> dims;
    if (input_rank == 1) {
        dims = {"x"};
    } else if (input_rank == 2) {
        dims = {"y", "x"};
    } else if (input_rank == 3) {
        dims = {"f", "y", "x"};
    } else if (input_rank == 4) {
        dims = {"b", "f", "y", "x"};
    } else if (input_rank == 5) {
        dims = {"b", "f", "z", "y", "x"};
    } else if (input_rank == 6) {
        dims = {"b", "f", "w", "z", "y", "x"};
    }
    return dims[axes] + " /= " + std::to_string(val) + ";";
}

std::string get_dims_order(const std::vector<int64_t>& order_idx) {
    auto get_order_idx = [](const std::vector<int64_t>& order_idx, int64_t dim_idx) {
        int loc = 0;
        for (auto idx : order_idx) {
            if (idx == dim_idx) {
                break;
            }
            loc += 1;
        }
        return loc;
    };

    std::string dims_order;
    if (order_idx.size() == 2) {
        const std::vector<std::string> dims2 = {"y", "x"};
        dims_order = "b,f,w,z," + dims2[get_order_idx(order_idx, 0)] + "," + dims2[get_order_idx(order_idx, 1)];
    } else if (order_idx.size() == 3) {
        const std::vector<std::string> dims3 = {"f", "y", "x"};
        dims_order = "b," + dims3[get_order_idx(order_idx, 0)] + ",w,z," + dims3[get_order_idx(order_idx, 1)] + "," + dims3[get_order_idx(order_idx, 2)];
    } else if (order_idx.size() == 4) {
        const std::vector<std::string> dims4 = {"b", "f", "y", "x"};
        dims_order = dims4[get_order_idx(order_idx, 0)] + "," + dims4[get_order_idx(order_idx, 1)] + ",w,z," + dims4[get_order_idx(order_idx, 2)] + "," +
                     dims4[get_order_idx(order_idx, 3)];
    } else if (order_idx.size() == 5) {
        const std::vector<std::string> dims5 = {"b", "f", "z", "y", "x"};
        dims_order = dims5[get_order_idx(order_idx, 0)] + "," + dims5[get_order_idx(order_idx, 1)] + ",w," + dims5[get_order_idx(order_idx, 2)] + "," +
                     dims5[get_order_idx(order_idx, 3)] + "," + dims5[get_order_idx(order_idx, 4)];
    } else if (order_idx.size() == 6) {
        const std::vector<std::string> dims6 = {"b", "f", "w", "z", "y", "x"};
        dims_order = dims6[get_order_idx(order_idx, 0)] + "," + dims6[get_order_idx(order_idx, 1)] + "," + dims6[get_order_idx(order_idx, 2)] + "," +
                     dims6[get_order_idx(order_idx, 3)] + "," + dims6[get_order_idx(order_idx, 4)] + "," + dims6[get_order_idx(order_idx, 5)];
    } else {
        dims_order = "b,f,w,z,y,x";
    }
    return dims_order;
}

size_t get_beam_table_id(const std::shared_ptr<const scaled_dot_product_attention>& primitive) {
    return primitive->input_size() - 1;
}

}  // namespace

std::pair<int64_t, int64_t> SDPABase::get_gqa_params(const kernel_impl_params& params) const {
    if (params.is_type<scaled_dot_product_attention>()) {
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
            if (order.empty())
                return pshape;

            auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
            for (size_t i = 0; i < order.size(); i++) {
                transposed_pshape[i] = pshape[order[i]];
            }
            return transposed_pshape;
        };

        const auto query_shape = transpose_pshape(params.get_input_layout(0).get_partial_shape(), desc->input_q_transpose_order);
        const auto key_shape = transpose_pshape(params.get_input_layout(1).get_partial_shape(), desc->input_k_transpose_order);
        const auto value_shape = transpose_pshape(params.get_input_layout(2).get_partial_shape(), desc->input_v_transpose_order);

        OPENVINO_ASSERT(key_shape == value_shape, "[GPU] The shapes of key and value inputs are expected to be equal");

        const auto num_heads_dim = 1;
        int64_t broadcast_axis = -1;
        int64_t group_size = -1;
        if (query_shape[num_heads_dim].is_static() && key_shape[num_heads_dim].is_static() && value_shape[num_heads_dim].is_static()) {
            if (query_shape[num_heads_dim].get_length() > key_shape[num_heads_dim].get_length()) {
                broadcast_axis = desc->input_k_transpose_order[num_heads_dim];
                group_size = query_shape[num_heads_dim].get_length() / key_shape[num_heads_dim].get_length();
            }
        }

        return {broadcast_axis, group_size};
    }
    if (params.is_type<paged_attention>()) {
        auto desc = params.typed_desc<paged_attention>();
        int64_t broadcast_axis = -1;
        int64_t group_size = -1;

        if (desc->heads_num != desc->kv_heads_num) {
            broadcast_axis = 1;
            group_size = desc->heads_num / desc->kv_heads_num;
        }

        return {broadcast_axis, group_size};
    }

    OPENVINO_THROW("[GPU] Wrong primitive type for get_gqa_params()");
}

JitConstants SDPABase::get_jit_constants(const kernel_impl_params& params) const {
    assert(params.is_type<scaled_dot_product_attention>() || params.is_type<paged_attention>());

    auto jit = make_base_jit_constants(params);
    auto [broadcast_axis, group_size] = get_gqa_params(params);
    if (broadcast_axis != -1) {
        jit.make("BROADCAST_GROUP_SIZE", group_size);
        jit.make("DO_BROADCAST_KEY_VALUE", get_broadcast_input_str(params.input_layouts[0].get_rank(), broadcast_axis, group_size));
    } else {
        jit.make("BROADCAST_GROUP_SIZE", 1);
    }

    if (params.is_type<scaled_dot_product_attention>()) {
        const auto& desc = params.typed_desc<scaled_dot_product_attention>();
        auto data_inputs_num = get_data_inputs_num(*desc);

        const size_t attn_mask_id = 3;
        const size_t scale_id = attn_mask_id + 1;

        jit.make("IS_CAUSAL", desc->is_causal);
        jit.make("HAS_ATTN_MASK_INPUT", data_inputs_num > attn_mask_id);
        jit.make("HAS_SCALE_INPUT", data_inputs_num > scale_id);

        jit.make("IS_KV_COMPRESSED", desc->is_kv_compressed);

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;

        if (desc->is_kv_compressed) {
            const auto& group_sizes = desc->quantization_attributes.group_sizes;
            const auto non_compressed_dims = std::count(group_sizes.begin(), group_sizes.end(), 1);

            const bool is_asym_quantization =
                desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
            const bool combined_scale_and_zp =
                desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

            const bool per_head_quantization = (group_sizes.size() - non_compressed_dims) == 1;
            jit.make("USE_ASYMMETRIC_QUANTIZATION", is_asym_quantization);
            jit.make("COMBINE_SCALES_AND_ZP", combined_scale_and_zp);
            jit.make("COMPRESSED_PER_HEAD", per_head_quantization);

            jit.add(make_layout_jit_constants("KEY_COMPRESSION_SCALE", params.input_layouts[data_inputs_num], in_offsets_map.at(data_inputs_num)));
            jit.add(make_layout_jit_constants("VALUE_COMPRESSION_SCALE", params.input_layouts[data_inputs_num + 1], in_offsets_map.at(data_inputs_num + 1)));

            if (is_asym_quantization && !combined_scale_and_zp) {
                jit.add(make_layout_jit_constants("KEY_COMPRESSION_ZP", params.input_layouts[data_inputs_num + 2], in_offsets_map.at(data_inputs_num + 2)));
                jit.add(make_layout_jit_constants("VALUE_COMPRESSION_ZP", params.input_layouts[data_inputs_num + 3], in_offsets_map.at(data_inputs_num + 3)));
            }
        }

        auto is_default_order = [](const std::vector<int64_t>& order) {
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i] != static_cast<int64_t>(i)) {
                    return false;
                }
            }
            return true;
        };

        // copy broadcast_axis as we can't capture structure binding in cpp17
        auto use_index_calc_func = [&, gqa_axis = broadcast_axis](const std::vector<int64_t> order, bool is_query = false) {
            if (!desc->input_q_transpose_order.empty() && !is_default_order(desc->input_q_transpose_order)) {
                return true;
            }

            if (gqa_axis != -1) {
                return true;
            }

            if (m_indirect && !is_query) {
                return true;
            }

            return false;
        };

        if (m_indirect) {
            const auto beam_table_id = get_beam_table_id(desc);
            jit.add(make_layout_jit_constants("BEAM_TABLE", params.input_layouts[beam_table_id], in_offsets_map.at(beam_table_id)));
        }

        if (use_index_calc_func(desc->input_q_transpose_order, true)) {
            jit.make("INPUT0_DIMS_ORDER", get_dims_order(desc->input_q_transpose_order));
        }

        if (use_index_calc_func(desc->input_k_transpose_order)) {
            jit.make("INPUT1_DIMS_ORDER", get_dims_order(desc->input_k_transpose_order));
        }

        if (use_index_calc_func(desc->input_v_transpose_order)) {
            jit.make("INPUT2_DIMS_ORDER", get_dims_order(desc->input_v_transpose_order));
        }

        LayoutJitter q_jitter(params.input_layouts[0], in_offsets_map.at(0));
        const auto num_heads = q_jitter.dim(get_transposed_channel(ChannelName::FEATURE, desc->input_q_transpose_order));
        jit.make("TARGET_SEQ_LEN", q_jitter.dim(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order)));
        jit.make("HEAD_SIZE", q_jitter.dim(get_transposed_channel(ChannelName::X, desc->input_q_transpose_order)));
        jit.make("NUM_HEADS", num_heads);

        LayoutJitter k_jitter(params.input_layouts[1], in_offsets_map.at(1));
        jit.make("SOURCE_SEQ_LEN", k_jitter.dim(get_transposed_channel(ChannelName::Y, desc->input_k_transpose_order)));
    } else if (params.is_type<paged_attention>()) {
        auto desc = params.typed_desc<paged_attention>();
        jit.make("IS_CAUSAL", true);
        jit.make("HEAD_SIZE", desc->head_size);
        jit.make("NUM_HEADS", desc->heads_num);
        jit.make("IS_KV_COMPRESSED", 0);

        if (desc->scale_val.has_value()) {
            jit.make("STATIC_SCALE_VALUE_INV", 1.0f / desc->scale_val.value());
            jit.make("STATIC_SCALE_VALUE", desc->scale_val.value());
        } else {
            jit.make("HAS_SCALE_INPUT", 1);
        }
    }

    return jit;
}

}  // namespace ov::intel_gpu::ocl
