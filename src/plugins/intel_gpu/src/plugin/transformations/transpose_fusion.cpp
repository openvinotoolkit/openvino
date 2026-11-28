// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_fusion.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <vector>

#include "graph/include/gemm_inst.h"
#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/vl_sdpa.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov::intel_gpu {

namespace {

bool is_valid_order(const std::vector<size_t>& target_order, bool is_output_transpose) {
    // Check valid input/output transpose order for onednn gemm primitive
    cldnn::format fmt_dummy = cldnn::format::bfyx;
    if (is_output_transpose) {
        return cldnn::typed_primitive_inst<cldnn::gemm>::is_fusable_permute_output_order_onednn(target_order, fmt_dummy);
    }
    return cldnn::typed_primitive_inst<cldnn::gemm>::is_fusable_permute_input_order_onednn(target_order, fmt_dummy);
}

bool has_optimized_version(const ov::Output<ov::Node>& output, bool supports_immad, bool is_output_transpose = false) {
    if (!output.get_element_type().is_real()) {
        return false;
    }

    if (output.get_partial_shape().is_static() && !supports_immad) {
        return false;
    }

    auto order_node = output.get_node()->get_input_node_shared_ptr(1);
    if (!ov::is_type<ov::op::v0::Constant>(order_node)) {
        return false;
    }

    auto transpose_order = ov::as_type_ptr<ov::op::v0::Constant>(order_node)->cast_vector<int64_t>();
    const auto expected_dims_num = 4;

    std::vector<size_t> order(std::begin(transpose_order), std::end(transpose_order));
    if (expected_dims_num > order.size()) {
        size_t orders_to_add = expected_dims_num - order.size();
        for (size_t i = 0; i < orders_to_add; ++i) {
            order.insert(order.begin(), i);
        }
        for (size_t i = orders_to_add; i < order.size(); ++i) {
            order[i] = order[i] + orders_to_add;
        }
    }

    return is_valid_order(order, is_output_transpose);
}
}  // namespace

TransposeFusion::TransposeFusion(bool supports_immad) {
    add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulMatcher>(supports_immad);
    add_matcher<TransposeSDPAMatcher>();
    add_matcher<TransposeVLSDPAMatcher>();
    add_matcher<TransposeSplitMatcher>();
}

TransposeVLSDPAMatcher::TransposeVLSDPAMatcher() {
    auto is_supported_type = type_matches_any(op::SDPA::get_supported_precisions());
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
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_supported_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_supported_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_supported_type);

    auto sdpa_m = wrap_type<ov::op::internal::VLSDPA>({transpose_q_m, transpose_k_m, transpose_v_m, input_cu_seqlens});

    // fuse output transpose into VLSDPA too
    auto transpose_o_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_o_m = wrap_type<ov::op::v1::Transpose>({sdpa_m, transpose_o_order_m}, is_supported_type);

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
            std::vector<int64_t> _order = transpose_order_const->cast_vector<int64_t>();

            // Allow any transposes without head_size dim position change
            if (_order.back() != static_cast<int64_t>(_order.size() - 1)) {
                return false;
            }

            auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            order = _order;

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q,
                                                     input_q_output_idx);
        }

        if (pattern_map.count(transpose_k_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);
        }

        if (pattern_map.count(transpose_v_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);
        }

        if (pattern_map.count(transpose_o_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_o_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_o_order_m).get_node_shared_ptr(),
                                                     order_output,
                                                     output_o_input_idx);
        }
        if (!can_fuse_transposes) {
            return false;
        }

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
    auto is_supported_type = type_matches_any(op::SDPA::get_supported_precisions());
    auto not_transpose = [is_supported_type](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && is_supported_type(output);
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_attn_mask = any_input(not_transpose);
    auto input_scale = any_input(not_transpose);
    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_supported_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_supported_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_supported_type);

    auto sdpa_in_q = std::make_shared<Or>(OutputVector{input_q_m, transpose_q_m});
    auto sdpa_in_k = std::make_shared<Or>(OutputVector{input_k_m, transpose_k_m});
    auto sdpa_in_v = std::make_shared<Or>(OutputVector{input_v_m, transpose_v_m});

    auto sdpa_without_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v});
    auto sdpa_with_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask});
    auto sdpa_with_attn_mask_and_scale_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask, input_scale});

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());
        auto gpu_sdpa = ov::as_type_ptr<op::SDPA>(m.get_match_root());

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        const auto causal_mask_alignment = gpu_sdpa ? gpu_sdpa->get_causal_mask_alignment()
                                                    : op::SDPA::CausalMaskAlignment::UPPER_LEFT;
        const auto output_type = gpu_sdpa ? gpu_sdpa->get_output_type() : ov::element::dynamic;

        auto order_q = op::SDPA::default_order(sdpa->get_input_partial_shape(0).size());
        auto order_k = op::SDPA::default_order(sdpa->get_input_partial_shape(1).size());
        auto order_v = op::SDPA::default_order(sdpa->get_input_partial_shape(2).size());
        auto order_output = op::SDPA::default_order(sdpa->get_output_partial_shape(0).size());
        if (gpu_sdpa) {
            if (!gpu_sdpa->get_input0_transpose_order().empty()) {
                order_q = gpu_sdpa->get_input0_transpose_order();
            }
            if (!gpu_sdpa->get_input1_transpose_order().empty()) {
                order_k = gpu_sdpa->get_input1_transpose_order();
            }
            if (!gpu_sdpa->get_input2_transpose_order().empty()) {
                order_v = gpu_sdpa->get_input2_transpose_order();
            }
            if (!gpu_sdpa->get_output_transpose_order().empty()) {
                order_output = gpu_sdpa->get_output_transpose_order();
            }
        }
        size_t input_q_output_idx = sdpa->get_input_source_output(0).get_index();
        size_t input_k_output_idx = sdpa->get_input_source_output(1).get_index();
        size_t input_v_output_idx = sdpa->get_input_source_output(2).get_index();

        auto process_transpose = [](const std::shared_ptr<Node>& transpose_node,
                                    const std::shared_ptr<Node>& transpose_order_const_node,
                                    std::vector<int64_t>& order,
                                    size_t& output_idx) {
            auto transpose_order_const = ov::as_type_ptr<ov::op::v0::Constant>(transpose_order_const_node);
            const auto transpose_order = transpose_order_const->cast_vector<int64_t>();

            // Allow any transposes without head_size dim position change
            if (transpose_order.back() != static_cast<int64_t>(transpose_order.size() - 1)) {
                return false;
            }

            if (order.size() != transpose_order.size()) {
                return false;
            }

            // composed[i] = transpose_order[order[i]]: keep the order already set on the SDPA
            std::vector<int64_t> composed(order.size());
            for (size_t i = 0; i < order.size(); ++i) {
                if (order[i] < 0 || static_cast<size_t>(order[i]) >= transpose_order.size()) {
                    return false;
                }
                composed[i] = transpose_order[static_cast<size_t>(order[i])];
            }
            order = composed;

            auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q,
                                                     input_q_output_idx);
        }

        if (pattern_map.count(transpose_k_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);
        }

        if (pattern_map.count(transpose_v_m) > 0) {
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);
        }

        if (!can_fuse_transposes) {
            return false;
        }

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);
        auto input_k = ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx);
        auto input_v = ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx);

        OutputVector inputs;
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx));

        if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            inputs.push_back(sdpa->get_input_source_output(3));
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            inputs.push_back(sdpa->get_input_source_output(3));
            inputs.push_back(sdpa->get_input_source_output(4));
        }

        auto sdpa_new =
            std::make_shared<op::SDPA>(inputs,
                                       sdpa->get_causal(),
                                       order_q,
                                       order_k,
                                       order_v,
                                       order_output,
                                       output_type,
                                       causal_mask_alignment,
                                       gpu_sdpa ? gpu_sdpa->get_sliding_window_size() : 0);

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
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && output.get_element_type().is_real();
    };

    auto transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad);
    };

    // Don't convert MatMul -> Gemm if no transpose input found as
    // CreateMatMulOp factory can now insert extra transpose which improves the performance
    auto matmul_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        auto* node = output.get_node();
        if (node->is_dynamic()) {
            return true;
        }

        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (ov::is_type<ov::op::v1::Transpose>(node->get_input_node_ptr(i))) {
                return true;
            }
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

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({matmul_in_a, matmul_in_b}, matmul_predicate);

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
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && output.get_element_type().is_real();
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

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({matmul_in_a, matmul_in_b}, consumers_count(1));
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

// TransposeSplitMatcher: Optimize Transpose+Split pattern from Qwen-VL Vision Merger
//
// Background:
// This pattern appears in Qwen-VL Vision Merger models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)
// where a combined QKV tensor
// with shape [-1, 3, num_head, head_size] needs to be split into separate Q, K, V tensors.
// The original framework uses Transpose to move the channel dimension to the front,
// then splits along that dimension.
//
// Original Pattern (from Qwen-VL Vision Merger):
//   Parameter[-1, 3, H, S] → Transpose[1,0,2,3] → [3, -1, H, S]
//     → Split(axis=0, num_splits=3) → 3x [1, -1, H, S]
//     → Reshape[0] → [-1, H, S] → RoPE → VLSDPA
//     → Reshape[1] → [-1, H, S] → RoPE ↗
//     → Reshape[2] → [-1, H, S] -------↗
//
// Optimized Pattern:
//   Parameter[-1, 3, H, S] → Split(axis=1, num_splits=3) → 3x [-1, 1, H, S]
//     → Reshape[0] → [-1, H, S] → RoPE → VLSDPA
//     → Reshape[1] → [-1, H, S] → RoPE ↗
//     → Reshape[2] → [-1, H, S] -------↗
//
// Benefits:
// - Eliminates unnecessary Transpose operation (saves memory bandwidth and kernel launch)
// - Produces functionally equivalent results: both patterns extract the same 3 slices
//   from the channel dimension, just with different intermediate shapes
// - Reshape operations following the Split automatically adapt to the new input shape
//
// Applicability:
// - Input shape: [-1, C, H, S] where C is strictly 3 (not dynamic)
// - Transpose order: [1, 0, 2, 3] (swaps first two dimensions only)
// - Split axis: 0 (after transpose, which corresponds to dim 1 before transpose)
// - num_splits: 3 (matching the channel count)
//
TransposeSplitMatcher::TransposeSplitMatcher() {
    // Pattern: Parameter[-1, 3, H, S] -> Transpose[3, -1, H, S] -> Split(axis=0) -> 3x[1, -1, H, S]
    // Optimize to: Parameter[-1, 3, H, S] -> Split(axis=1) -> 3x[-1, 1, H, S]

    auto input_m = any_input();
    auto transpose_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({input_m, transpose_order_m});
    auto split_axis_m = wrap_type<ov::op::v0::Constant>();
    auto split_m = wrap_type<ov::op::v1::Split>({transpose_m, split_axis_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto split = ov::as_type_ptr<ov::op::v1::Split>(pattern_map.at(split_m).get_node_shared_ptr());
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_m).get_node_shared_ptr());
        auto input_node = pattern_map.at(input_m).get_node_shared_ptr();

        if (!split || !transpose || transformation_callback(split)) {
            return false;
        }

        // Get transpose order
        auto transpose_order_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_order_m).get_node_shared_ptr());
        if (!transpose_order_const) {
            return false;
        }
        auto transpose_order = transpose_order_const->cast_vector<int64_t>();

        // Get split axis
        auto split_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(split_axis_m).get_node_shared_ptr());
        if (!split_axis_const) {
            return false;
        }
        auto split_axis_vec = split_axis_const->cast_vector<int64_t>();
        if (split_axis_vec.size() != 1) {
            return false;
        }
        int64_t split_axis = split_axis_vec[0];

        // Get input shape
        auto input_pshape = input_node->get_output_partial_shape(0);
        if (input_pshape.rank().is_dynamic() || input_pshape.rank().get_length() != 4) {
            return false;
        }

        // Check conditions for the optimization:
        // 1. Input shape: [-1, 3, H, S] where dim[1] is strictly 3
        // 2. Transpose order: [1, 0, 2, 3] (swaps first two dimensions)
        // 3. Split axis: 0 (after transpose, splitting the "3" dimension)
        // 4. num_splits: 3

        // Condition 1: Check dim[1] is strictly 3
        if (input_pshape[1].is_dynamic() || input_pshape[1].get_length() != 3) {
            return false;
        }

        // Condition 2: Check transpose order is [1, 0, 2, 3]
        std::vector<int64_t> expected_transpose_order = {1, 0, 2, 3};
        if (transpose_order != expected_transpose_order) {
            return false;
        }

        // Condition 3: Check split axis is 0 (after transpose)
        if (split_axis != 0) {
            return false;
        }

        // Condition 4: Check num_splits is 3
        if (split->get_num_splits() != 3) {
            return false;
        }

        // Create new Split that operates directly on axis=1 of the input (before transpose)
        // This produces 3 outputs of shape [-1, 1, H, S] instead of [1, -1, H, S]
        auto new_split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto new_split = std::make_shared<ov::op::v1::Split>(input_node, new_split_axis, split->get_num_splits());
        ov::copy_runtime_info(m.get_matched_nodes(), new_split);
        ov::replace_node(split, new_split);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(split_m, "TransposeSplitMatcher");
    this->register_matcher(matcher, callback);
}

namespace {

std::optional<int64_t> get_scalar_const_i64(const ov::Output<ov::Node>& output) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    if (!constant || ov::shape_size(constant->get_shape()) != 1) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>()[0];
}

bool is_scalar_const(const ov::Output<ov::Node>& output) {
    return ov::is_type<ov::op::v0::Constant>(output.get_node()) && ov::shape_size(output.get_shape()) == 1;
}

// True if every consumer of `output` is a Q/K/V input of an SDPA, either directly or
// through a Multiply by a scalar constant whose consumers are all Q/K/V inputs of an SDPA.
bool feeds_only_sdpa_qkv(const ov::Output<ov::Node>& output) {
    const auto is_sdpa_qkv = [](const ov::Input<ov::Node>& in) {
        return ov::is_type<ov::op::v13::ScaledDotProductAttention>(in.get_node()) && in.get_index() < 3;
    };
    const auto& consumers = output.get_target_inputs();
    const auto is_sdpa_qkv_or_scaled = [&](const ov::Input<ov::Node>& in) {
        if (is_sdpa_qkv(in)) {
            return true;
        }
        auto* mul = ov::as_type<ov::op::v1::Multiply>(in.get_node());
        if (mul == nullptr || !is_scalar_const(mul->input_value(1 - in.get_index()))) {
            return false;
        }
        const auto& mul_consumers = mul->get_output_target_inputs(0);
        return !mul_consumers.empty() && std::all_of(mul_consumers.begin(), mul_consumers.end(), is_sdpa_qkv);
    };
    return !consumers.empty() && std::all_of(consumers.begin(), consumers.end(), is_sdpa_qkv_or_scaled);
}

// True if `node` only removes the axis `axis` of its input, which is statically 1: Squeeze(axis),
// or a Reshape whose static output shape is the static input shape without that axis.
bool removes_unit_axis(const std::shared_ptr<ov::Node>& node, size_t axis, int64_t rank) {
    if (ov::is_type<ov::op::util::SqueezeBase>(node)) {
        if (node->get_input_size() != 2) {
            return false;
        }
        auto squeeze_axis = get_scalar_const_i64(node->input_value(1));
        if (!squeeze_axis || (*squeeze_axis < 0 ? *squeeze_axis + rank : *squeeze_axis) != static_cast<int64_t>(axis)) {
            return false;
        }
        // Squeeze keeps a selected axis whose size is not 1, so the slice must be statically unit.
        const auto& dim = node->get_input_partial_shape(0)[axis];
        return dim.is_static() && dim.get_length() == 1;
    }
    if (ov::is_type<ov::op::v1::Reshape>(node)) {
        const auto& in = node->get_input_partial_shape(0);
        const auto& out = node->get_output_partial_shape(0);
        if (in.is_dynamic() || out.is_dynamic() || in.size() != out.size() + 1 || in[axis] != 1) {
            return false;
        }
        auto expected = in.to_shape();
        expected.erase(expected.begin() + axis);
        return expected == out.to_shape();
    }
    return false;
}

// Order of the transpose that remains after removing axis `removed_pos` of `order`.
std::vector<int64_t> drop_axis_from_order(const std::vector<int64_t>& order, size_t removed_pos) {
    const int64_t removed_axis = order[removed_pos];
    std::vector<int64_t> reduced;
    for (size_t i = 0; i < order.size(); ++i) {
        if (i != removed_pos) {
            reduced.push_back(order[i] > removed_axis ? order[i] - 1 : order[i]);
        }
    }
    return reduced;
}

bool keeps_last_axis(const std::vector<int64_t>& order) {
    return !order.empty() && order.back() == static_cast<int64_t>(order.size()) - 1;
}

// Rewrites one Transpose whose consumers unpack it along one axis into SDPA inputs; see the header.
bool sink_transpose_through_split(const std::shared_ptr<ov::op::v1::Transpose>& transpose) {
    const auto order_const = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
    const auto input = transpose->input_value(0);
    if (!order_const || input.get_partial_shape().rank().is_dynamic()) {
        return false;
    }
    const auto rank = input.get_partial_shape().rank().get_length();
    const auto order = order_const->cast_vector<int64_t>();
    if (rank < 2 || static_cast<int64_t>(order.size()) != rank) {
        return false;
    }
    const auto normalize = [rank](int64_t axis) -> std::optional<size_t> {
        axis = axis < 0 ? axis + rank : axis;
        if (axis < 0 || axis >= rank) {
            return std::nullopt;
        }
        return static_cast<size_t>(axis);
    };

    const auto consumers = transpose->get_output_target_inputs(0);
    if (consumers.empty()) {
        return false;
    }

    // Each branch is a node whose output is "transposed input with axis a removed at index i".
    // Complete unpacks (all slices along a) are rewritten as one split; anything else as
    // per-branch Gathers.
    std::optional<size_t> a;
    std::vector<std::shared_ptr<ov::Node>> branches;  // indexed by slice position
    std::vector<std::pair<std::shared_ptr<ov::op::util::GatherBase>, size_t>> partial_gathers;
    std::vector<std::shared_ptr<ov::Node>> replaced = {transpose};

    auto split_node = consumers.begin()->get_node()->shared_from_this();
    if (consumers.size() == 1 && (ov::is_type<ov::op::v1::Split>(split_node) || ov::is_type<ov::op::v1::VariadicSplit>(split_node))) {
        const auto split_axis = get_scalar_const_i64(split_node->input_value(1));
        a = split_axis ? normalize(*split_axis) : std::nullopt;
        if (!a) {
            return false;
        }
        for (const auto& out : split_node->outputs()) {
            const auto& targets = out.get_target_inputs();
            if (targets.size() != 1) {
                return false;
            }
            auto squeeze = targets.begin()->get_node()->shared_from_this();
            if (!removes_unit_axis(squeeze, *a, rank) || !feeds_only_sdpa_qkv(squeeze->output(0))) {
                return false;
            }
            branches.push_back(squeeze);
        }
        replaced.push_back(split_node);
    } else {
        for (const auto& in : consumers) {
            auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(in.get_node()->shared_from_this());
            if (!gather || in.get_index() != 0 || gather->get_batch_dims() != 0 ||
                gather->get_input_partial_shape(1).rank().is_dynamic() || gather->get_input_partial_shape(1).rank().get_length() != 0 ||
                !feeds_only_sdpa_qkv(gather->output(0))) {
                return false;
            }
            const auto index = get_scalar_const_i64(gather->input_value(1));
            const auto gather_axis = get_scalar_const_i64(gather->input_value(2));
            if (!index || !gather_axis || !normalize(*gather_axis)) {
                return false;
            }
            const size_t axis = *normalize(*gather_axis);
            const auto& dim = transpose->get_output_partial_shape(0)[axis];
            if (dim.is_dynamic() || *index < -dim.get_length() || *index >= dim.get_length()) {
                return false;
            }
            partial_gathers.emplace_back(gather, *index < 0 ? *index + dim.get_length() : *index);
            if (!a) {
                a = axis;
            } else if (*a != axis) {
                a = std::numeric_limits<size_t>::max();
            }
        }
        // All gathers on one axis, each slice taken exactly once: equivalent to a split.
        if (*a != std::numeric_limits<size_t>::max() &&
            static_cast<int64_t>(partial_gathers.size()) == transpose->get_output_partial_shape(0)[*a].get_length()) {
            branches.assign(partial_gathers.size(), nullptr);
            for (const auto& [gather, index] : partial_gathers) {
                if (branches[index]) {
                    break;
                }
                branches[index] = gather;
            }
            if (std::all_of(branches.begin(), branches.end(), [](const auto& b) { return b != nullptr; })) {
                partial_gathers.clear();
            } else {
                branches.clear();
            }
        }
    }

    const auto make_transpose = [](const ov::Output<ov::Node>& in, const std::vector<int64_t>& reduced_order) {
        // One order constant per Transpose: TransposeSDPAMatcher requires a single consumer.
        return std::make_shared<ov::op::v1::Transpose>(
            in, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduced_order.size()}, reduced_order));
    };

    if (!branches.empty()) {
        const auto reduced_order = drop_axis_from_order(order, *a);
        if (!keeps_last_axis(reduced_order)) {
            return false;
        }
        const int64_t new_axis = order[*a];
        const auto& in_pshape = input.get_partial_shape();
        const auto num_slices = static_cast<int64_t>(branches.size());
        auto new_axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {new_axis});

        // Prefer merging the split axis with the next one ([.., n, H, ..] -> [.., n*H, ..]) so that
        // each slice keeps the rank of the SDPA input and no Squeeze is needed: the split outputs
        // can then be consumed in place. This needs static dims from the split axis on; dims
        // before it are kept via special_zero.
        bool can_merge = new_axis + 1 < rank;
        for (int64_t i = new_axis; can_merge && i < rank; ++i) {
            can_merge = in_pshape[i].is_static();
        }
        std::shared_ptr<ov::Node> new_split;
        if (can_merge) {
            std::vector<int64_t> merged_shape(new_axis, 0);
            merged_shape.push_back(in_pshape[new_axis].get_length() * in_pshape[new_axis + 1].get_length());
            for (int64_t i = new_axis + 2; i < rank; ++i) {
                merged_shape.push_back(in_pshape[i].get_length());
            }
            auto merged = std::make_shared<ov::op::v1::Reshape>(
                input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{merged_shape.size()}, merged_shape), true);
            new_split = std::make_shared<ov::op::v1::Split>(merged, new_axis_const, num_slices);
            ov::copy_runtime_info(replaced, merged);
        } else {
            new_split = std::make_shared<ov::op::v1::Split>(input, new_axis_const, num_slices);
        }
        new_split->set_friendly_name(replaced.back()->get_friendly_name());
        ov::copy_runtime_info(replaced, new_split);

        for (size_t i = 0; i < branches.size(); ++i) {
            ov::Output<ov::Node> slice = new_split->output(i);
            if (!can_merge) {
                slice = std::make_shared<ov::op::v0::Squeeze>(slice, new_axis_const);
                ov::copy_runtime_info({transpose, branches[i]}, slice.get_node_shared_ptr());
            }
            auto new_transpose = make_transpose(slice, reduced_order);
            new_transpose->set_friendly_name(branches[i]->get_friendly_name());
            ov::copy_runtime_info({transpose, branches[i]}, new_transpose);
            branches[i]->output(0).replace(new_transpose->output(0));
        }
        return true;
    }

    if (partial_gathers.empty()) {
        return false;
    }
    for (const auto& [gather, index] : partial_gathers) {
        const size_t axis = *normalize(*get_scalar_const_i64(gather->input_value(2)));
        if (!keeps_last_axis(drop_axis_from_order(order, axis))) {
            return false;
        }
    }
    for (const auto& [gather, index] : partial_gathers) {
        const size_t axis = *normalize(*get_scalar_const_i64(gather->input_value(2)));
        auto new_gather = std::make_shared<ov::op::v8::Gather>(
            input, gather->input_value(1), ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {order[axis]}));
        auto new_transpose = make_transpose(new_gather, drop_axis_from_order(order, axis));
        new_gather->set_friendly_name(gather->get_friendly_name() + "/gather");
        new_transpose->set_friendly_name(gather->get_friendly_name());
        ov::copy_runtime_info({transpose, gather}, {new_gather, new_transpose});
        gather->output(0).replace(new_transpose->output(0));
    }
    return true;
}

}  // namespace

bool SinkTransposeThroughSplitForSDPA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // A ModelPass: the rewrite replaces the consumers (Split/Squeeze/Gather branches) of the Transpose.
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node);
        if (transpose && !transformation_callback(transpose)) {
            changed |= sink_transpose_through_split(transpose);
        }
    }
    return changed;
}

}  // namespace ov::intel_gpu
