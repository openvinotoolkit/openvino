// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_fuse_transpose_reshape.hpp"

#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"

/*
 * Description: SDPA fuse transpose and reshape.
 *           Original pattern                            Fused pattern
 *
 *  input1         input2       input3
 *     |             |             |
 * q_reshape     k_reshape     v_reshap
 *     |             |             |                         (qkv transpose and reshape's orders)
 * q_transpose  k_transpose   v_transpose                                     |
 *         \         |        /                      input1  input2  input3   |
 *          \        |       /                          \      |       /      /
 *       ScaledDotProductAttention   --------->        SDPAWithTransposeReshape
 *                   |                                         |
 *              out_transpose                                  |
 *                   |                                       output
 *               out_reshpae
 *                   |
 *                 output
 */

using namespace ov;
using namespace ov::pass::pattern;

intel_cpu::SDPAFuseTransposeReshape::SDPAFuseTransposeReshape() {
    MATCHER_SCOPE(SDPAFuseTransposeReshape);

    auto q_reshape_node = wrap_type<op::v1::Reshape>({any_input(), any_input()});
    auto k_reshape_node = wrap_type<op::v1::Reshape>({any_input(), any_input()});
    auto v_reshape_node = wrap_type<op::v1::Reshape>({any_input(), any_input()});

    auto q_transpose_order_node = wrap_type<op::v0::Constant>();
    auto k_transpose_order_node = wrap_type<op::v0::Constant>();
    auto v_transpose_order_node = wrap_type<op::v0::Constant>();
    auto q_transpose_node = wrap_type<op::v1::Transpose>({q_reshape_node, q_transpose_order_node});
    auto k_transpose_node = wrap_type<op::v1::Transpose>({k_reshape_node, k_transpose_order_node});
    auto v_transpose_node = wrap_type<op::v1::Transpose>({v_reshape_node, v_transpose_order_node});

    auto sdpa_node =
        wrap_type<op::v13::ScaledDotProductAttention>({q_transpose_node, k_transpose_node, v_transpose_node});

    auto out_transpose_order_node = wrap_type<op::v0::Constant>();
    auto out_transpose_node = wrap_type<op::v1::Transpose>({sdpa_node, out_transpose_order_node});
    auto out_reshape_node = wrap_type<op::v1::Reshape>({out_transpose_node, wrap_type<op::v0::Constant>()});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto sdpa = as_type_ptr<op::v13::ScaledDotProductAttention>(pattern_map.at(sdpa_node).get_node_shared_ptr());
        if (sdpa == nullptr || transformation_callback(sdpa)) {
            return false;
        }

        // Order=[0, 2, 1, 3]
        auto is_expected_transpose = [&](std::shared_ptr<op::v1::Transpose>& transpose) {
            if (transpose) {
                const auto orders = as_type_ptr<op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
                return orders && (std::vector<int32_t>({0, 2, 1, 3}) == orders->cast_vector<int32_t>());
            }
            return false;
        };

        // Reshape [B,L,H*S] -> [B,L,H,S]
        auto is_expected_reshape = [&](std::shared_ptr<op::v1::Reshape>& reshape_node, bool reverse = false) {
            if (reshape_node) {
                auto inp_shape = reshape_node->get_input_partial_shape(0);
                auto outp_shape = reshape_node->get_output_partial_shape(0);
                // Expect shape: [?, ?, val]
                auto check_dim_3 = [](ov::PartialShape shape) {
                    return shape.rank().is_static() && shape.rank() == 3 && shape[2].is_static();
                };
                // Expect shape: [?, ?, val, val]
                auto check_dim_4 = [](ov::PartialShape shape) {
                    return shape.rank().is_static() && shape.rank() == 4 && shape[2].is_static() &&
                           shape[3].is_static();
                };

                if (reverse) {
                    return check_dim_4(inp_shape) && check_dim_3(outp_shape) &&
                           (outp_shape[2] == inp_shape[2] * inp_shape[3]);
                } else {
                    return check_dim_3(inp_shape) && check_dim_4(outp_shape) &&
                           (inp_shape[2] == outp_shape[2] * outp_shape[3]);
                }
            }
            return false;
        };

        // Pattern: Reshape->Transpose->SDPA
        auto q_reshape = as_type_ptr<op::v1::Reshape>(pattern_map.at(q_reshape_node).get_node_shared_ptr());
        auto k_reshape = as_type_ptr<op::v1::Reshape>(pattern_map.at(k_reshape_node).get_node_shared_ptr());
        auto v_reshape = as_type_ptr<op::v1::Reshape>(pattern_map.at(v_reshape_node).get_node_shared_ptr());

        if (!(is_expected_reshape(q_reshape) && is_expected_reshape(k_reshape) && is_expected_reshape(v_reshape))) {
            return false;
        }
        // K,V Reshape's order should be same node.
        auto k_reshape_order = as_type_ptr<op::v0::Constant>(k_reshape->get_input_node_shared_ptr(1));
        auto v_reshape_order = as_type_ptr<op::v0::Constant>(v_reshape->get_input_node_shared_ptr(1));
        if (k_reshape_order && v_reshape_order) {
            if (k_reshape_order->cast_vector<int32_t>() != v_reshape_order->cast_vector<int32_t>()) {
                return false;
            }
        } else if (k_reshape->get_input_node_shared_ptr(1) != v_reshape->get_input_node_shared_ptr(1)) {
            return false;
        }

        std::shared_ptr<op::v1::Transpose> qkv_transpose[3] = {};
        std::shared_ptr<op::v0::Constant> qkv_transpose_order[3] = {};
        qkv_transpose[0] = as_type_ptr<op::v1::Transpose>(pattern_map.at(q_transpose_node).get_node_shared_ptr());
        qkv_transpose[1] = as_type_ptr<op::v1::Transpose>(pattern_map.at(k_transpose_node).get_node_shared_ptr());
        qkv_transpose[2] = as_type_ptr<op::v1::Transpose>(pattern_map.at(v_transpose_node).get_node_shared_ptr());
        qkv_transpose_order[0] = as_type_ptr<op::v0::Constant>(pattern_map.at(q_transpose_order_node).get_node_shared_ptr());
        qkv_transpose_order[1] = as_type_ptr<op::v0::Constant>(pattern_map.at(k_transpose_order_node).get_node_shared_ptr());
        qkv_transpose_order[2] = as_type_ptr<op::v0::Constant>(pattern_map.at(v_transpose_order_node).get_node_shared_ptr());
        auto out_tranpose = as_type_ptr<op::v1::Transpose>(pattern_map.at(out_transpose_node).get_node_shared_ptr());
        auto out_transpose_order = as_type_ptr<op::v0::Constant>(pattern_map.at(out_transpose_order_node).get_node_shared_ptr());

        if (!(is_expected_transpose(qkv_transpose[0]) && is_expected_transpose(qkv_transpose[1]) &&
              is_expected_transpose(qkv_transpose[2]))) {
            return false;
        }
        if (!is_expected_transpose(out_tranpose)) {
            return false;
        }

        auto out_reshape = as_type_ptr<op::v1::Reshape>(pattern_map.at(out_reshape_node).get_node_shared_ptr());
        if (!is_expected_reshape(out_reshape, true)) {
            return false;
        }

        OutputVector args = {q_reshape->get_input_node_shared_ptr(0),
                             k_reshape->get_input_node_shared_ptr(0),
                             v_reshape->get_input_node_shared_ptr(0)};

        // Config
        intel_cpu::SDPAWithTransposeReshape::Config config;
        config.is_causal = sdpa->get_causal();
        config.fuse_concat = false;
        config.output_BLHxS = true;

        // Config::permute_axes
        const auto& permute_q = qkv_transpose_order[0]->cast_vector<int32_t>();
        config.permute_axes.resize(permute_q.size());
        for (size_t i = 0; i < permute_q.size(); i++) {
            config.permute_axes[i] = static_cast<size_t>(permute_q[i]);
        }

        // Config::order_HS
        config.order_HS.resize(2);
        auto reshape_out_shape = q_reshape->get_output_partial_shape(0).get_min_shape();  // [?,?,H,S]
        config.order_HS[0] = reshape_out_shape[2];
        config.order_HS[1] = reshape_out_shape[3];
        config.input_BLHxS = true;

        auto new_sdpa = std::make_shared<intel_cpu::SDPAWithTransposeReshape>(args, config);
        new_sdpa->set_friendly_name(sdpa->get_friendly_name() + "/fused_reshape_transpose");
        NodeVector replaced_nodes = {q_reshape,
                                     k_reshape,
                                     v_reshape,
                                     qkv_transpose[0],
                                     qkv_transpose[1],
                                     qkv_transpose[2],
                                     sdpa,
                                     out_tranpose,
                                     out_reshape};
        copy_runtime_info(replaced_nodes, new_sdpa);
        ov::replace_node(out_reshape, new_sdpa);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(out_reshape_node, matcher_name);
    register_matcher(m, callback);
}
