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

        // auto order_to_string = [](const std::vector<int64_t>& order) {
        //         std::ostringstream oss;
        //         oss << "[";
        //         for (size_t i = 0; i < order.size(); ++i) {
        //             oss << order[i];
        //             if (i < order.size() - 1) {
        //                 oss << ", ";
        //             }
        //         }
        //         oss << "]";
        //         return oss.str();
        // };
        // std::cout << "SDPABase::get_gqa_params:" << std::endl;
        // std::cout << " \tinput Q transpose order: " << order_to_string(desc->input_q_transpose_order) << "\n";
        // std::cout << " \tinput K transpose order: " << order_to_string(desc->input_k_transpose_order) << "\n";
        // std::cout << " \tinput V transpose order: " << order_to_string(desc->input_v_transpose_order) << "\n";
        // std::cout << " \toutput transpose order: " << order_to_string(desc->output_transpose_order) << "\n";

        // std::cout << "\tquery shape: " << query_shape.to_string() << ", desc->input_q_transpose_order = " << desc->input_q_transpose_order[0] << ", "
        //           << desc->input_q_transpose_order[1] << "," << desc->input_q_transpose_order[2] << std::endl;
        // std::cout << "\tkey shape: " << key_shape.to_string() << ", desc->input_k_transpose_order = " << desc->input_k_transpose_order[0] << ", "
        //           << desc->input_k_transpose_order[1] << "," << desc->input_k_transpose_order[2] << std::endl;
        // std::cout << "\tvalue shape: " << value_shape.to_string() << ", desc->input_v_transpose_order = " << desc->input_v_transpose_order[0] << ", "
        //           << desc->input_v_transpose_order[1] << "," << desc->input_v_transpose_order[2] << std::endl;

        const auto num_heads_dim = 1;
        int64_t broadcast_axis = -1;
        int64_t group_size = -1;
        if (query_shape[num_heads_dim].is_static() && key_shape[num_heads_dim].is_static() && value_shape[num_heads_dim].is_static()) {
            if (query_shape[num_heads_dim].get_length() > key_shape[num_heads_dim].get_length()) {
                broadcast_axis = desc->input_k_transpose_order[num_heads_dim];
                group_size = query_shape[num_heads_dim].get_length() / key_shape[num_heads_dim].get_length();
            }
        }

        // std::cout << "SDPABase::get_gqa_params: done" << std::endl;
        return {broadcast_axis, group_size};
    }
    if (params.is_type<paged_attention>()) {
        // For micro kernel shared between SDPAs and Paged Attention
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

sdpa_configuration SDPABase::get_sdpa_configuration(const kernel_impl_params& impl_param,
                                                    const std::vector<int64_t>& input_q_transpose_order,
                                                    const std::vector<int64_t>& input_k_transpose_order,
                                                    const std::vector<int64_t>& input_v_transpose_order) {
    sdpa_configuration config;

    auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
        if (order.empty())
            return pshape;

        auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
        for (size_t i = 0; i < order.size(); i++) {
            transposed_pshape[i] = pshape[order[i]];
        }
        return transposed_pshape;
    };

    const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
    const auto query_shape = transpose_pshape(impl_param.get_input_layout(0).get_partial_shape(), input_q_transpose_order);
    const auto key_shape = transpose_pshape(impl_param.get_input_layout(1).get_partial_shape(), input_k_transpose_order);
    const auto value_shape = transpose_pshape(impl_param.get_input_layout(2).get_partial_shape(), input_v_transpose_order);

    const auto num_heads_dim = 1;
    if (query_shape[num_heads_dim].is_static() && key_shape[num_heads_dim].is_static() && value_shape[num_heads_dim].is_static()) {
        if (query_shape[num_heads_dim].get_length() > key_shape[num_heads_dim].get_length()) {
            config.broadcast_axis = desc->input_k_transpose_order[num_heads_dim];
            config.kv_group_size = query_shape[num_heads_dim].get_length() / key_shape[num_heads_dim].get_length();
        }
    }

    if (query_shape[query_shape.size() - 1].is_static())
        config.k_head_size = query_shape[query_shape.size() - 1].get_length();

    if (value_shape[value_shape.size() - 1].is_static())
        config.v_head_size = value_shape[value_shape.size() - 1].get_length();

    // std::cout << "SDPABase::get_sdpa_configuration: value_shape = " << value_shape.to_string() << ",  v_head_size = " << config.v_head_size << std::endl;

    config.is_causal = desc->is_causal;

    if (desc->scale_val.has_value()) {
        config.has_const_scale_val = true;
        config.scale_val = desc->scale_val.value();
    } else {
        config.has_const_scale_val = false;
    }

    if (desc->attn_mask_val.has_value()) {
        config.has_const_attn_mask_val = true;
        config.attn_mask_val = desc->attn_mask_val.value();
    } else {
        config.has_const_attn_mask_val = false;
    }

    if (desc->is_kv_compressed) {
        const auto& group_sizes = desc->quantization_attributes.group_sizes;
        const auto non_compressed_dims = std::count(group_sizes.begin(), group_sizes.end(), 1);

        config.per_head_quantization = (group_sizes.size() - non_compressed_dims) == 1;
        config.is_kv_compressed = desc->is_kv_compressed;
        config.use_asymmetric_quantization = desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        config.combine_scales_and_zp = desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
    }

    // figure out sdpa input number: QKV + attention_mask + scale, exclude: beam table, key compression scale/zp, value compression scale/zp
    config.input_num = get_data_inputs_num(*desc);

    return config;
}

JitConstants SDPABase::get_jit_constants(const kernel_impl_params& params) const {
    assert(params.is_type<scaled_dot_product_attention>() || params.is_type<paged_attention>());

    auto jit = make_base_jit_constants(params);
    auto [broadcast_axis, group_size] = get_gqa_params(params);
    std::cout << "SDPABase::get_jit_constants: broadcast_axis = " << broadcast_axis << ", group_size = " << group_size << std::endl;
    if (broadcast_axis != -1) {
        jit.make("BROADCAST_GROUP_SIZE", group_size);
        jit.make("DO_BROADCAST_KEY_VALUE", get_broadcast_input_str(params.input_layouts[0].get_rank(), broadcast_axis, group_size));
    } else {
        jit.make("BROADCAST_GROUP_SIZE", 1);
    }

    if (params.is_type<scaled_dot_product_attention>()) {
        const auto& desc = params.typed_desc<scaled_dot_product_attention>();
        auto data_inputs_num = get_data_inputs_num(*desc);

        std::cout << "SDPABase::get_jit_constants: -----1" << std::endl;
        const size_t attn_mask_id = 3;
        // const size_t scale_id = attn_mask_id + 1;

        jit.make("IS_CAUSAL", desc->is_causal);
        if (desc->attn_mask_val.has_value()) {
            jit.make("STATIC_SCALAR_ATTN_MASK_VALUE", desc->attn_mask_val.value());
            jit.make("HAS_ATTN_MASK_INPUT", 0);
        } else {
            jit.make("HAS_ATTN_MASK_INPUT", data_inputs_num > attn_mask_id);
        }

        // jit.make("HAS_SCALE_INPUT", data_inputs_num > scale_id);
        jit.make("IS_KV_COMPRESSED", desc->is_kv_compressed);
         std::cout << "SDPABase::get_jit_constants: desc->is_kv_compressed = " << desc->is_kv_compressed << std::endl;

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
            if (!order.empty() && !is_default_order(order)) {
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

        std::cout << "SDPABase::get_jit_constants: -----3" << std::endl;
        if (m_indirect) {
            const auto beam_table_id = get_beam_table_id(desc);
            jit.add(make_layout_jit_constants("BEAM_TABLE", params.input_layouts[beam_table_id], in_offsets_map.at(beam_table_id)));
        }

        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
        auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

        std::cout << "SDPABase::get_jit_constants: -----4" << std::endl;
        if (use_index_calc_func(extended_input_q_transpose_order, true)) {
            jit.make("INPUT0_DIMS_ORDER", get_dims_order(extended_input_q_transpose_order));
        }

        if (use_index_calc_func(extended_input_k_transpose_order)) {
            jit.make("INPUT1_DIMS_ORDER", get_dims_order(extended_input_k_transpose_order));
        }

        if (use_index_calc_func(extended_input_v_transpose_order)) {
            jit.make("INPUT2_DIMS_ORDER", get_dims_order(extended_input_v_transpose_order));
        }
        std::cout << "SDPABase::get_jit_constants: -----5" << std::endl;

        auto updated_params = static_canonicalize_shapes(params);
        LayoutJitter q_jitter(updated_params.input_layouts[0], in_offsets_map.at(0));
        jit.make("TARGET_SEQ_LEN", q_jitter.dim(get_transposed_channel(ChannelName::Y, extended_input_q_transpose_order)));
        // jit.make("HEAD_SIZE", q_jitter.dim(get_transposed_channel(ChannelName::X, desc->input_q_transpose_order)));
        // jit.make("NUM_HEADS", q_jitter.dim(get_transposed_channel(ChannelName::FEATURE, desc->input_q_transpose_order)));

        // std::cout << "SDPABase::get_jit_constants: -----6" << std::endl;
        LayoutJitter k_jitter(updated_params.input_layouts[1], in_offsets_map.at(1));
        jit.make("SOURCE_SEQ_LEN", k_jitter.dim(get_transposed_channel(ChannelName::Y, extended_input_q_transpose_order)));
        // jit.make("NUM_KV_HEADS", k_jitter.dim(get_transposed_channel(ChannelName::FEATURE, desc->input_k_transpose_order)));
        // jit.make("K_HEAD_SIZE", k_jitter.dim(get_transposed_channel(ChannelName::X, desc->input_k_transpose_order)));
        // std::cout << "SDPABase::get_jit_constants: -----7" << std::endl;

        // LayoutJitter v_jitter(params.input_layouts[2], in_offsets_map.at(2));
        // jit.make("V_HEAD_SIZE", v_jitter.dim(get_transposed_channel(ChannelName::X, desc->input_v_transpose_order)));

        auto order_to_string = [](const std::vector<int64_t>& order) {
                std::ostringstream oss;
                oss << "[";
                for (size_t i = 0; i < order.size(); ++i) {
                    oss << order[i];
                    if (i < order.size() - 1) {
                        oss << ", ";
                    }
                }
                oss << "]";
                return oss.str();
        };
        std::cout << " input Q transpose order: " << order_to_string(desc->input_q_transpose_order)
                  << "->" <<order_to_string(extended_input_q_transpose_order) << "\n";
        std::cout << " input K transpose order: " << order_to_string(desc->input_k_transpose_order)
                  << "->" <<order_to_string(extended_input_k_transpose_order) << "\n";
        std::cout << " input V transpose order: " << order_to_string(desc->input_v_transpose_order)
                  << "->" <<order_to_string(extended_input_v_transpose_order) << "\n";
        std::cout << " output transpose order: " << order_to_string(desc->output_transpose_order)
                  << "->" <<order_to_string(extended_output_transpose_order) << "\n";

        std::cout << " query: " << params.get_input_layout(0).to_string() << std::endl;
        std::cout << " key: " << params.get_input_layout(1).to_string() << std::endl;
        std::cout << " value: " << params.get_input_layout(2).to_string() << std::endl;
        // std::cout << " query: " << params.get_input_layout(0).to_string() << std::endl;

        const auto q_head_size = get_head_size(params.get_input_layout(0), extended_input_q_transpose_order);
        const auto q_seq_len = get_seq_length(params.get_input_layout(0), extended_input_q_transpose_order);
        const auto q_num_head = get_num_heads(params.get_input_layout(0), extended_input_q_transpose_order);
        const auto k_head_size = get_head_size(params.get_input_layout(1), extended_input_k_transpose_order);
        const auto k_seq_len = get_seq_length(params.get_input_layout(1), extended_input_k_transpose_order);
        const auto k_num_head = get_num_heads(params.get_input_layout(1), extended_input_k_transpose_order);
        const auto v_head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);
        //jit.make("TARGET_SEQ_LEN", q_seq_len);
        jit.make("HEAD_SIZE", q_head_size);
        jit.make("NUM_HEADS", q_num_head);
        //jit.make("SOURCE_SEQ_LEN", k_seq_len);
        jit.make("K_HEAD_SIZE", k_head_size);
        jit.make("NUM_KV_HEADS", k_num_head);
        jit.make("V_HEAD_SIZE", v_head_size);

        std::cout << "q_seq_len = " << q_seq_len << ", q_num_head = " << q_num_head << ", q_head_size = " << q_head_size << std::endl;
        std::cout << "k_seq_len = " << k_seq_len << ", k_num_head = " << k_num_head << ", k_head_size = " << k_head_size << std::endl;
        std::cout << "v_head_size = " << v_head_size << std::endl;

        std::cout << "SDPABase::get_jit_constants: -----7" << std::endl;
        // auto v_head_size = v_jitter.dim(get_transposed_channel(ChannelName::X, desc->input_v_transpose_order));
        // std::cout << "SDPABase::get_jit_constants: v_head_size = " << v_head_size << std::endl;

    } else if (params.is_type<paged_attention>()) {
        // For micro/sdpa kernel shared between SDPAs and Paged Attention
        auto desc = params.typed_desc<paged_attention>();
        jit.make("IS_CAUSAL", true);
        jit.make("K_HEAD_SIZE", desc->k_head_size);
        jit.make("V_HEAD_SIZE", desc->v_head_size);
        jit.make("NUM_HEADS", desc->heads_num);
        jit.make("KV_NUM_HEADS", desc->kv_heads_num);

        if (desc->scale_val.has_value()) {
            jit.make("STATIC_SCALE_VALUE_INV", 1.0f / desc->scale_val.value());
            jit.make("STATIC_SCALE_VALUE", desc->scale_val.value());
        } else {
            jit.make("HAS_SCALE_INPUT", 1);
        }
    }

    return jit;
}

bool SDPABase::requires_shape_canonicalization(const kernel_impl_params& impl_params) {
    auto extend_output = impl_params.output_layouts[0].get_partial_shape().size() < 4;
    auto extend_attn_mask = false;
    // According to SDPA specification, attention mask should have 2-dimensions or more or empty
    size_t attn_mask_idx = 3;
    if (impl_params.input_layouts.size() > attn_mask_idx) {
        const auto& attn_mask_shape = impl_params.get_input_layout(attn_mask_idx).get_partial_shape();
        extend_attn_mask = attn_mask_shape.size() != 0 && attn_mask_shape.size() < 4;
    }

    return extend_output || extend_attn_mask;
}

kernel_impl_params SDPABase::static_canonicalize_shapes(const kernel_impl_params& impl_params) {
    auto updated_impl_params = impl_params;

    auto extend_pshape_to_rank_in_num_heads_dim = [](ov::PartialShape pshape, size_t rank = 4) {
        if (pshape.size() == rank) {
            return pshape;
        }
        const size_t num_heads_dim = 1;
        pshape.insert(pshape.begin() + num_heads_dim, ov::Dimension(1));
        return pshape;
    };

    const auto attn_mask_idx = 3;
    if (updated_impl_params.input_layouts.size() > attn_mask_idx) {
        const auto attn_mask_shape = updated_impl_params.input_layouts[attn_mask_idx].get_partial_shape();
        updated_impl_params.input_layouts[attn_mask_idx].set_partial_shape(extend_shape_to_rank_from_begin(attn_mask_shape));
    }

    // For scale of 1D tensor or attention_mask of empty shape, use extend_shape_to_rank_from_end as before
    for (auto& input_layout : updated_impl_params.input_layouts) {
        std::cout << "static_canonicalize_shapes: input_layout.get_partial_shape() = " << input_layout.get_partial_shape().to_string() << std::endl;
        input_layout.set_partial_shape(input_layout.get_partial_shape().size() <= 1 ? extend_shape_to_rank_from_end(input_layout.get_partial_shape())
                                                                                    : extend_pshape_to_rank_in_num_heads_dim(input_layout.get_partial_shape()));
    }

    auto& output_layout = updated_impl_params.output_layouts[0];
    output_layout.set_partial_shape(extend_pshape_to_rank_in_num_heads_dim(output_layout.get_partial_shape()));

    // const auto& desc = impl_params.typed_desc<scaled_dot_product_attention>();
    // auto order_to_string = [](const std::vector<int64_t>& order) {
    //         std::ostringstream oss;
    //         oss << "[";
    //         for (size_t i = 0; i < order.size(); ++i) {
    //             oss << order[i];
    //             if (i < order.size() - 1) {
    //                 oss << ", ";
    //             }
    //         }
    //         oss << "]";
    //         return oss.str();
    // };
    // std::cout << " input Q transpose order: " << order_to_string(desc->input_q_transpose_order) << "\n";
    // std::cout << " input K transpose order: " << order_to_string(desc->input_k_transpose_order) << "\n";
    // std::cout << " input V transpose order: " << order_to_string(desc->input_v_transpose_order) << "\n";
    // std::cout << " output transpose order: " << order_to_string(desc->output_transpose_order) << "\n";

    // auto& updated_desc = reinterpret_cast<scaled_dot_product_attention&>(updated_impl_params);
    // auto extend_order_in_num_heads_dim = [](const std::vector<int64_t>& order, size_t rank = 4) {
    //     if (order.size() == rank) {
    //         return order;
    //     }

    //     std::vector<int64_t> extended_order(rank, 0);
    //     const size_t num_heads_dim = 1;
    //     // For 3D dimension, extend it to 4D by adding 1 for num_heads_dim
    //     for (size_t i = 0, j = 0; i < rank; ++i) {
    //         if (i == num_heads_dim) {
    //             extended_order[num_heads_dim] = 1;
    //         } else {
    //             extended_order[i] = (static_cast<size_t>(order[j]) < num_heads_dim) ? order[j] : order[j] + 1;
    //             j++;
    //         }
    //     }
    //     return extended_order;
    // };
    // updated_desc.input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
    // updated_desc.input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
    // updated_desc.input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
    // updated_desc.output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

    // std::cout << "Extended input Q transpose order: " << order_to_string(desc->input_q_transpose_order) << "->"
    //         << order_to_string(updated_desc.input_q_transpose_order) << "\n";
    // std::cout << "Extended input K transpose order: " << order_to_string(desc->input_k_transpose_order) << "->"
    //         << order_to_string(updated_desc.input_k_transpose_order) << "\n";
    // std::cout << "Extended input V transpose order: " << order_to_string(desc->input_v_transpose_order) << "->"
    //         << order_to_string(updated_desc.input_v_transpose_order) << "\n";
    // std::cout << "Extended output transpose order: " << order_to_string(desc->output_transpose_order) << "->"
    //         << order_to_string(updated_desc.output_transpose_order) << "\n";

    return updated_impl_params;
}

void SDPAImplBase::update(cldnn::primitive_inst& inst, const RuntimeParams& impl_params) {
    auto new_impl_params = SDPABase::requires_shape_canonicalization(impl_params) ? SDPABase::static_canonicalize_shapes(impl_params) : impl_params;
    // std::cout << "SDPAImplBase::update: requires_shape_canonicalization(impl_params) = "
    // << SDPABase::requires_shape_canonicalization(impl_params) << std::endl;
    // std::cout << "\t impl_params.input_layouts[0] = " << impl_params.input_layouts[0].to_string() << std::endl;
    // std::cout << "\t new_impl_params.input_layouts[0] = " << new_impl_params.input_layouts[0].to_string() << std::endl;
    inst.update_shape_info_tensor(new_impl_params);
}

}  // namespace ov::intel_gpu::ocl
