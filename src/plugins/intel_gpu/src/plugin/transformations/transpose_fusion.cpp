// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_fusion.hpp"

#include <iostream>
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
#include "openvino/op/Convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
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
    std::cout << " valid order = " << is_valid_order(order, is_output_transpose) << std::endl;
    return is_valid_order(order, is_output_transpose);
}
}  // namespace

TransposeFusion::TransposeFusion(bool supports_immad) {
    add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulMatcher>(supports_immad);
    add_matcher<TransposeSDPAMatcher>();
    add_matcher<TransposeVLSDPAMatcher>();
    add_matcher<TransposeConv1x1TransposeMatcher>(supports_immad);
}

TransposeVLSDPAMatcher::TransposeVLSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
        case ov::element::f16:
        case ov::element::f32:
            return true;
        default:
            return false;
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

    auto sdpa_m = wrap_type<ov::op::internal::VLSDPA>({transpose_q_m, transpose_k_m, transpose_v_m, input_cu_seqlens});

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
            std::vector<int64_t> _order = transpose_order_const->cast_vector<int64_t>();

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
                                                     order_q,
                                                     input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);

        if (pattern_map.count(transpose_o_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_o_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_o_order_m).get_node_shared_ptr(),
                                                     order_output,
                                                     output_o_input_idx);
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
        case ov::element::f32:
            return true;
        default:
            return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && is_fp_type(output);
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

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q,
                                                     input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);

        if (!can_fuse_transposes)
            return false;

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

        auto sdpa_new = std::make_shared<op::SDPA>(inputs, sdpa->get_causal(), order_q, order_k, order_v, order_output);

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

// Weight is 9216 x 3072 x 1 x 1
// Activation 1 x 1 x 64 x 3072

// *********************** First solution ***********************************
// If we have [1 x 1 x 9216 x 3072] * [1 x 1 x 3072 x 64] = [1 x 1 x 9216 x 64]
// For the above, we need to apply [3,2,1,0] transpose order on weights. With that, I got an error from gemDNN Kenrl that [3,2,1,0] is not
// fusable transpose

//*********************** Second Solution *************************************
// Reshape weight from [9216 x 3072 x 1 x 1] to [9216 x 3072]
// [9216 x 3072] * [1 x 1 x 3072 x 64] = [ 1 x 1 x 9216 x 64]
/* Add Reshape on weight and apply transpose on Activation */
// The solution does not work, Matmul/Gemm have FP16 activations, int4 weights. So, we need a kernel to convert int4 weight to FP16 weight which is missing
// [GPU] Could not find a suitable kernel for convert:Convert_13 params raw string:
// INT4_BFYX_v1_p0_0_v1_p0_0_v3072_p0_0_v9216_p0_0;F16_BFYX_v1_p0_0_v1_p0_0_v3072_p0_0_v9216_p0_0

// *********************** Thirs Solution *************************************
// Reshape weight
// Let the original int4 weights goes through Matmul
// run Matmul/Gemm with int4 W and FP16 Activations
// The above solution works fine if we make changes in gemm_oneDNN.hpp to allow u4 input datatypes
//However, the oneDNN kernel is not optimized. One of the main thing is attr-fpmath:f16 is not set to True. Seems that everything upcast to PF32
//oneDNN Verbose
//onednn_verbose,v1,primitive,exec,gpu,matmul,ocl:ref:any,undef,src:f16::blocked:abcd::f0 wei:u4::blocked:abdc::f0 dst:f16::blocked:abcd::f0,attr-scratchpad:user,,1x1x64x3072:1x1x3072x9216,418.188
// The kernel is not optimized. It takes 480 ms to run and fall back to some ocl kernel.
// I run qwen2-0-5b-instruct-merged-int4.onnx which has int4 weights and the oneDNN verbose looks like below
//nednn_verbose,v1,primitive,exec,gpu,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:7:f16:32x1 attr-zero-points:wei:0:u8 attr-post-ops:binary_add:f16:4,,1x1x896:1x896x896,0.9225

/*
TransposeConv1x1TransposeMatcher::TransposeConv1x1TransposeMatcher() {
    auto static_rank_gt_1 = [](const ov::Output<ov::Node>& output) {
        const auto& r = output.get_partial_shape().rank();
        return r.is_static() && r.get_length() > 1;
    };
    auto weights_path = [&static_rank_gt_1](const ov::Output<ov::Node>& output) {
        const auto& pshape = output.get_partial_shape();
        return ov::op::util::is_on_constant_path(output) && static_rank_gt_1(output) && pshape.is_static() &&
               std::count_if(pshape.begin(), pshape.end(), [](const ov::Dimension& x) {
                   return x == 1;
               }) == 2;
    };

    auto first_input_m = ov::pass::pattern::any_input();
    auto transpose_a_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({first_input_m, transpose_a_order_m});

    auto weights_m = ov::pass::pattern::any_input(weights_path);  // weights
    auto conv1x1_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({transpose_activations_m, weights_m});

    auto transpose_c_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_output_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({conv1x1_m, transpose_c_order_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv1x1 = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map.at(conv1x1_m).get_node_shared_ptr());
        auto transpose_output = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_output_m).get_node_shared_ptr());
        auto transpose_activations = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_activations_m).get_node_shared_ptr());
        if (!conv1x1 || transformation_callback(conv1x1)) {
            return false;
        }

        auto weight = pattern_map.at(weights_m).get_node_shared_ptr();
        auto activation = pattern_map.at(first_input_m).get_node_shared_ptr();
        auto order_activation = op::Gemm::default_order(conv1x1->get_input_partial_shape(0).size()); //activation
        auto order_weight = op::Gemm::default_order(conv1x1->get_input_partial_shape(1).size()); // weight




        for (auto i = 0; i < order_activation.size(); i++) {
            std::cout << order_activation[i] << std::endl;
        }



        order_weight[0] = 3;
        order_weight[1] = 2;
        order_weight[2] = 1;
        order_weight[3] = 0;

        order_activation[0] = 0;
        order_activation[1] = 1;
        order_activation[2] = 2;
        order_activation[3] = 3;

       auto order_c = order_activation;
       order_c[0] = 0;
       order_c[1] = 1;
       order_c[2] = 2;
       order_c[3] = 3;
       //ad reshape after weight
       std::vector<int> values_reshape_b;
       auto shape_b = weight->get_input_partial_shape(0);
       for (auto i = 0; i < shape_b.size(); i++)
           if (shape_b.to_shape()[i] != 1) {
               values_reshape_b.push_back(shape_b.to_shape()[i]);
           }


       auto reshape_weight_const = ov::op::v0::Constant::create(element::i32, Shape{2}, values_reshape_b);  //{9216, 3072});
       auto Reshape_weight = std::make_shared<ov::op::v1::Reshape>(weight, reshape_weight_const, false);
       MatcherPass::register_new_node(Reshape_weight);
       Reshape_weight->set_friendly_name(weight->get_friendly_name() + "_Reshape_weight");
       ov::disable_constant_folding(Reshape_weight);



       //auto gemm = std::make_shared<op::Gemm>(weight, transpose_activations, order_weight, order_activation, order_c);
       auto gemm = std::make_shared<ov::op::v0::MatMul>(Reshape_weight, activation, false, true);
        gemm->set_friendly_name(conv1x1->get_friendly_name());
        ov::copy_runtime_info(conv1x1, gemm);
        ov::replace_node(conv1x1, gemm);
        ov::replace_node(transpose_output, gemm);
        //ov::replace_node(transpose_activations, gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_output_m, "TransposeMatMulTransposeMatcher");
    this->register_matcher(m, callback);
}

*/
// Third solution code

TransposeConv1x1TransposeMatcher::TransposeConv1x1TransposeMatcher(bool supports_immad) {
    auto static_rank_gt_1 = [](const ov::Output<ov::Node>& output) {
        const auto& r = output.get_partial_shape().rank();
        return r.is_static() && r.get_length() > 1;
    };
    auto weights_path = [&static_rank_gt_1](const ov::Output<ov::Node>& output) {
        const auto& pshape = output.get_partial_shape();
        return ov::op::util::is_on_constant_path(output) && static_rank_gt_1(output) && pshape.is_static() &&
               std::count_if(pshape.begin(), pshape.end(), [](const ov::Dimension& x) {
                   return x == 1;
               }) == 2;
    };
    auto input_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, false);
    };
    auto output_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, true);
    };

    
    auto first_input_m = ov::pass::pattern::any_input();
    auto transpose_a_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({first_input_m, transpose_a_order_m}); //, input_transpose_predicate);

    auto weights_m = ov::pass::pattern::any_input(weights_path);  // weights
    auto weight_convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights_m});
    auto weight_subtract_m = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({weight_convert_m, ov::pass::pattern::any_input()});
    auto weight_mult_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({weight_subtract_m, ov::pass::pattern::any_input()});
    auto conv1x1_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({transpose_activations_m, weight_mult_m});

    auto transpose_c_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_output_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({conv1x1_m, transpose_c_order_m}); //, output_transpose_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv1x1 = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map.at(conv1x1_m).get_node_shared_ptr());
        auto transpose_output = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_output_m).get_node_shared_ptr());
        auto transpose_activations = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_activations_m).get_node_shared_ptr());
        if (!conv1x1 || transformation_callback(conv1x1)) {
            return false;
        }

        auto weight = pattern_map.at(weights_m).get_node_shared_ptr();
        auto activation = pattern_map.at(first_input_m).get_node_shared_ptr();
        
        auto order_activation = op::Gemm::default_order(conv1x1->get_input_partial_shape(0).size());  // activation

        order_activation[0] = 0;
        order_activation[1] = 1;
        order_activation[2] = 2;
        order_activation[3] = 3;
        

        auto order_c = order_activation;
        order_c[0] = 0;
        order_c[1] = 1;
        order_c[2] = 2;
        order_c[3] = 3;
        
        // add reshape after weight 9216 x 3072 x 1 x --> 9216 x 3072
        std::vector<int> values_reshape_b;
        auto shape_b = weight->get_output_partial_shape(0);
        for (auto i = 0; i < shape_b.size(); i++)
            if (shape_b.to_shape()[i] != 1) {
                values_reshape_b.push_back(shape_b.to_shape()[i]);
            }

         

        auto reshape_weight_const = ov::op::v0::Constant::create(element::i32, Shape{2}, values_reshape_b);  //{9216, 3072});
        auto Reshape_weight = std::make_shared<ov::op::v1::Reshape>(weight, reshape_weight_const, false);
        MatcherPass::register_new_node(Reshape_weight);
        Reshape_weight->set_friendly_name(weight->get_friendly_name() + "_Reshape_weight");
        ov::disable_constant_folding(Reshape_weight);
        auto order_weight = op::Gemm::default_order(Reshape_weight->get_output_partial_shape(0).size());  // weight

        order_weight[0] = 1;
        order_weight[1] = 0;
        

        auto gemm = std::make_shared<op::Gemm>(activation, Reshape_weight, order_activation, order_weight, order_c);
        gemm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(m.get_match_root(), gemm);

   
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_output_m, "TransposeMatMulTransposeMatcher");
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

}  // namespace ov::intel_gpu