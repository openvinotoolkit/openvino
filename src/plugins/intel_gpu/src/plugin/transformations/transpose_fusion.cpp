// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "transpose_fusion.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "graph/include/gemm_inst.h"

#include "ov_ops/vl_sdpa.hpp"
#include "ov_ops/dynamic_quantize.hpp"

#include <iostream>
#include <limits>
#include <vector>
#include <ostream>

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov::intel_gpu {

namespace {

bool is_valid_order(const std::vector<size_t>& target_order, bool is_output_transpose) {
    // Check valid input/output transpose order for onednn gemm primitive
    cldnn::format fmt_dummy = cldnn::format::bfyx;
    if (is_output_transpose) {
        return cldnn::typed_primitive_inst<cldnn::gemm>::is_fusable_permute_output_order_onednn(target_order, fmt_dummy);
    } else {
        return cldnn::typed_primitive_inst<cldnn::gemm>::is_fusable_permute_input_order_onednn(target_order, fmt_dummy);
    }
}

bool has_optimized_version(const ov::Output<ov::Node>& output, bool supports_immad, bool is_output_transpose = false) {
    if (!output.get_element_type().is_real())
        return false;

    if (output.get_partial_shape().is_static() && !supports_immad)
        return false;

    auto order_node = output.get_node()->get_input_node_shared_ptr(1);
    if (!ov::is_type<ov::op::v0::Constant>(order_node))
        return false;

    auto transpose_order = ov::as_type_ptr<ov::op::v0::Constant>(order_node)->cast_vector<int64_t>();
    const auto expected_dims_num = 4;

    std::vector<size_t> order(std::begin(transpose_order), std::end(transpose_order));
    if (expected_dims_num > order.size()) {
        size_t orders_to_add = expected_dims_num - order.size();
        for (size_t i = 0; i < orders_to_add; ++i)
            order.insert(order.begin(), i);
        for (size_t i = orders_to_add; i < order.size(); ++i)
            order[i] = order[i] + orders_to_add;
    }

    return is_valid_order(order, is_output_transpose);
}

// Returns true for integer element types used to store a quantized KV cache.
bool is_quantized_kv_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 ||
           type == ov::element::i4 || type == ov::element::u4;
}

// Derive DynamicQuantize group sizes from the quantized data shape and the (broadcastable) scale shape:
// an axis where the scale extent is 1 while the data extent is > 1 is fully grouped (max), every other
// axis keeps a group size of 1. Falls back to grouping the last axis when shapes are unknown.
std::vector<uint64_t> compute_kv_group_sizes(const ov::PartialShape& data_ps, const ov::PartialShape& scale_ps) {
    if (data_ps.rank().is_dynamic()) {
        return {};
    }
    const size_t rank = data_ps.rank().get_length();
    std::vector<uint64_t> group_sizes(rank, 1);
    if (scale_ps.rank().is_static() && static_cast<size_t>(scale_ps.rank().get_length()) == rank) {
        for (size_t i = 0; i < rank; ++i) {
            const bool scale_is_one = scale_ps[i].is_static() && scale_ps[i].get_length() == 1;
            const bool data_is_one = data_ps[i].is_static() && data_ps[i].get_length() == 1;
            if (scale_is_one && !data_is_one) {
                group_sizes[i] = std::numeric_limits<uint64_t>::max();
            }
        }
    } else if (rank > 0) {
        group_sizes[rank - 1] = std::numeric_limits<uint64_t>::max();
    }
    return group_sizes;
}
}  // namespace

TransposeFusion::TransposeFusion(bool supports_immad) {
    add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulMatcher>(supports_immad);
    add_matcher<TransposeSDPAMatcher>();
    add_matcher<TransposeVLSDPAMatcher>();
}

TransposeVLSDPAMatcher::TransposeVLSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr;
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_cu_seqlens = any_input(not_transpose);

    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_fp_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_fp_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_fp_type);

    auto sdpa_m = wrap_type<ov::op::internal::VLSDPA>({ transpose_q_m, transpose_k_m, transpose_v_m, input_cu_seqlens });

    // fuse output transpose into VLSDPA too
    auto transpose_o_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_o_m = wrap_type<ov::op::v1::Transpose>({sdpa_m, transpose_o_order_m}, is_fp_type);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa = ov::as_type_ptr<ov::op::internal::VLSDPA>(pattern_map.at(sdpa_m).get_node_shared_ptr());

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        auto order_q = op::SDPA::default_order(sdpa->get_input_partial_shape(0).size());
        auto order_k = op::SDPA::default_order(sdpa->get_input_partial_shape(1).size());
        auto order_v = op::SDPA::default_order(sdpa->get_input_partial_shape(2).size());
        auto order_output = op::SDPA::default_order(sdpa->get_output_partial_shape(0).size());
        size_t input_q_output_idx = sdpa->get_input_source_output(0).get_index();
        size_t input_k_output_idx = sdpa->get_input_source_output(1).get_index();
        size_t input_v_output_idx = sdpa->get_input_source_output(2).get_index();
        size_t output_o_input_idx = sdpa->get_input_source_output(2).get_index();

        auto process_transpose = [](const std::shared_ptr<Node>& transpose_node,
                                    const std::shared_ptr<Node>& transpose_order_const_node,
                                    std::vector<int64_t>& order,
                                    size_t& output_idx) {
            auto transpose_order_const = ov::as_type_ptr<ov::op::v0::Constant>(transpose_order_const_node);
            std::vector<int64_t>_order = transpose_order_const->cast_vector<int64_t>();

            // Allow any transposes without head_size dim position change
            if (_order.back() != static_cast<int64_t>(_order.size() - 1))
                return false;

            auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            order = _order;

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q, input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k, input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v, input_v_output_idx);

        if (pattern_map.count(transpose_o_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_o_m).get_node_shared_ptr(),
                                                    pattern_map.at(transpose_o_order_m).get_node_shared_ptr(),
                                                    order_output, output_o_input_idx);
        if (!can_fuse_transposes)
            return false;

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);
        auto input_k = ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx);
        auto input_v = ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx);

        OutputVector inputs;
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx));
        inputs.push_back(sdpa->get_input_source_output(3));
        auto sdpa_new = std::make_shared<ov::op::internal::VLSDPA>(inputs, order_q, order_k, order_v, order_output);

        auto transpose_o = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_o_m).get_node_shared_ptr());
        ov::replace_node(transpose_o, sdpa_new);

        sdpa_new->set_friendly_name(transpose_o->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_o_m, "TransposeVLSDPAMatcher");
    this->register_matcher(m, callback);
}

TransposeSDPAMatcher::TransposeSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && is_fp_type(output);
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_attn_mask = any_input(not_transpose);
    auto input_scale = any_input(not_transpose);
    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_fp_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_fp_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_fp_type);

    // KV-cache dequantization: Multiply(optional Subtract(Convert(quant), zp), scale) [-> optional Reshape].
    // Matches the canonical low-precision dequantization sub-graph (see low_precision_dequantize) emitted for
    // an int8/int4 KV cache, so the quantized cache and its scale/zero-point can be folded into op::SDPA.
    auto k_quant_m = any_input();
    auto k_convert_m = wrap_type<ov::op::v0::Convert>({k_quant_m});
    auto k_zp_m = any_input();
    auto k_sub_m = ov::pass::pattern::optional<ov::op::v1::Subtract>({k_convert_m, k_zp_m});
    auto k_scale_m = any_input();
    auto k_mul_m = wrap_type<ov::op::v1::Multiply>({k_sub_m, k_scale_m});
    auto k_dequant_m = ov::pass::pattern::optional<ov::op::v1::Reshape>({k_mul_m, any_input()});

    auto v_quant_m = any_input();
    auto v_convert_m = wrap_type<ov::op::v0::Convert>({v_quant_m});
    auto v_zp_m = any_input();
    auto v_sub_m = ov::pass::pattern::optional<ov::op::v1::Subtract>({v_convert_m, v_zp_m});
    auto v_scale_m = any_input();
    auto v_mul_m = wrap_type<ov::op::v1::Multiply>({v_sub_m, v_scale_m});
    auto v_dequant_m = ov::pass::pattern::optional<ov::op::v1::Reshape>({v_mul_m, any_input()});

    // Dequantization branch is tried first so a quantized KV cache is recognized before the plain-float fallback.
    auto sdpa_in_q = std::make_shared<Or>(OutputVector{input_q_m, transpose_q_m});
    auto sdpa_in_k = std::make_shared<Or>(OutputVector{k_dequant_m, transpose_k_m, input_k_m});
    auto sdpa_in_v = std::make_shared<Or>(OutputVector{v_dequant_m, transpose_v_m, input_v_m});

    auto sdpa_without_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v });
    auto sdpa_with_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask });
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask, input_scale });

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        auto order_q = op::SDPA::default_order(sdpa->get_input_partial_shape(0).size());
        auto order_k = op::SDPA::default_order(sdpa->get_input_partial_shape(1).size());
        auto order_v = op::SDPA::default_order(sdpa->get_input_partial_shape(2).size());
        auto order_output = op::SDPA::default_order(sdpa->get_output_partial_shape(0).size());
        size_t input_q_output_idx = sdpa->get_input_source_output(0).get_index();
        size_t input_k_output_idx = sdpa->get_input_source_output(1).get_index();
        size_t input_v_output_idx = sdpa->get_input_source_output(2).get_index();

        auto process_transpose = [](const std::shared_ptr<Node>& transpose_node,
                                    const std::shared_ptr<Node>& transpose_order_const_node,
                                    std::vector<int64_t>& order,
                                    size_t& output_idx) {
            auto transpose_order_const = ov::as_type_ptr<ov::op::v0::Constant>(transpose_order_const_node);

            order = transpose_order_const->cast_vector<int64_t>();
            // Allow any transposes without head_size dim position change
            if (order.back() != static_cast<int64_t>(order.size() - 1))
                return false;

            auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            return true;
        };

        // A KV input is quantized when its dequantization sub-graph matched and the source is an integer cache.
        const bool k_quantized =
            pattern_map.count(k_mul_m) > 0 && is_quantized_kv_type(pattern_map.at(k_quant_m).get_element_type());
        const bool v_quantized =
            pattern_map.count(v_mul_m) > 0 && is_quantized_kv_type(pattern_map.at(v_quant_m).get_element_type());
        // Both K and V must be quantized to build a compressed SDPA (it carries one scale per KV input).
        const bool kv_compressed = k_quantized && v_quantized;

        static const auto compressedkv = []() {
            const auto txt = std::getenv("compressedkv");
            return !(txt && txt == std::string_view("false"));
        }();
        
        if (kv_compressed && !compressedkv)
            return false;

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q, input_q_output_idx);

        // Transpose fusion is only applied to the plain-float KV path (the quantized cache keeps identity order).
        if (!k_quantized && pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k, input_k_output_idx);

        if (!v_quantized && pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v, input_v_output_idx);

        if (!can_fuse_transposes)
            return false;

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);

        // K/V source: the raw quantized cache when compressing, the transpose source on the fused-float path,
        // otherwise the SDPA input as-is (covers the dequant-but-not-compressed fallback).
        auto select_kv_input = [&](bool compressed,
                                   const std::shared_ptr<ov::Node>& quant_pattern,
                                   const std::shared_ptr<ov::Node>& float_pattern,
                                   size_t float_output_idx,
                                   size_t sdpa_input_idx) -> ov::Output<ov::Node> {
            if (compressed) {
                return pattern_map.at(quant_pattern);
            }
            if (pattern_map.count(float_pattern) > 0) {
                return ov::Output<Node>(pattern_map.at(float_pattern).get_node_shared_ptr(), float_output_idx);
            }
            return sdpa->input_value(sdpa_input_idx);
        };

        auto input_k = select_kv_input(kv_compressed, k_quant_m, input_k_m, input_k_output_idx, 1);
        auto input_v = select_kv_input(kv_compressed, v_quant_m, input_v_m, input_v_output_idx, 2);

        OutputVector inputs = {input_q, input_k, input_v};

        if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            inputs.push_back(sdpa->get_input_source_output(3));
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            inputs.push_back(sdpa->get_input_source_output(3));
            inputs.push_back(sdpa->get_input_source_output(4));
        }

        std::shared_ptr<ov::Node> sdpa_new;
        if (kv_compressed) {
            const auto k_scale = pattern_map.at(k_scale_m);
            const auto v_scale = pattern_map.at(v_scale_m);
            const bool asymmetric = pattern_map.count(k_sub_m) > 0 && pattern_map.count(v_sub_m) > 0;

            op::SDPA::QuantizationAttribute config;
            config.quantization_type = asymmetric ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
                                                  : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
            config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
            config.quantization_dt = pattern_map.at(k_quant_m).get_element_type();
            config.scale_dt = k_scale.get_element_type();
            config.group_sizes =
                compute_kv_group_sizes(pattern_map.at(k_quant_m).get_partial_shape(), k_scale.get_partial_shape());
            config.scales_zp_output_order = std::vector<uint64_t>{0, 1, 2, 3};
            if (asymmetric) {
                config.zp_dt = pattern_map.at(k_zp_m).get_element_type();
            }

            // Compression inputs follow the data (and optional mask/scale) inputs: K/V scales, then K/V zero points.
            inputs.push_back(k_scale);
            inputs.push_back(v_scale);
            if (asymmetric &&
                config.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
                inputs.push_back(pattern_map.at(k_zp_m));
                inputs.push_back(pattern_map.at(v_zp_m));
            }

            sdpa_new = std::make_shared<op::SDPA>(inputs,
                                                  sdpa->get_causal(),
                                                  order_q,
                                                  order_k,
                                                  order_v,
                                                  order_output,
                                                  config);
        } else {
            sdpa_new = std::make_shared<op::SDPA>(inputs, sdpa->get_causal(), order_q, order_k, order_v, order_output);
        }

        sdpa_new->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        ov::replace_node(sdpa, sdpa_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "TransposeSDPAMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulMatcher::TransposeMatMulMatcher(bool supports_immad) {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && output.get_element_type().is_real();
    };

    auto transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad);
    };

    // Don't convert MatMul -> Gemm if no transpose input found as
    // CreateMatMulOp factory can now insert extra transpose which improves the performance
    auto matmul_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        auto node = output.get_node();
        if (node->is_dynamic())
            return true;

        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (ov::is_type<ov::op::v1::Transpose>(node->get_input_node_ptr(i)))
                return true;
        }

        return false;
    };

    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, transpose_predicate);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, transpose_predicate);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b }, matmul_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto users = matmul->get_output_target_inputs(0);
        if (users.size() == 1 && ov::as_type<ov::op::v1::Transpose>(users.begin()->get_node()) != nullptr) {
            return false;
        }

        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = op::Gemm::default_order(matmul->get_output_partial_shape(0).size());
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
            input_b_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_b() && order_b.size() > 1) {
            std::swap(*(order_b.end() - 1), *(order_b.end() - 2));
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(matmul, gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "TransposeMatMulMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulTransposeMatcher::TransposeMatMulTransposeMatcher(bool supports_immad) {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && output.get_element_type().is_real();
    };
    auto input_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, false);
    };
    auto output_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, true);
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, input_transpose_predicate);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, input_transpose_predicate);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b }, consumers_count(1));
    auto transpose_c_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_c_m = wrap_type<ov::op::v1::Transpose>({matmul_m, transpose_c_order_m}, output_transpose_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto tranpose_c_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_c_order_m).get_node_shared_ptr());
        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = tranpose_c_order->cast_vector<int64_t>();
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
            input_b_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_b() && order_b.size() > 1) {
            std::swap(*(order_b.end() - 1), *(order_b.end() - 2));
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(m.get_match_root(), gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_c_m, "TransposeMatMulTransposeMatcher");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
