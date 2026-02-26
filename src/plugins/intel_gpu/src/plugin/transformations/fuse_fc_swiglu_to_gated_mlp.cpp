// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_fc_swiglu_to_gated_mlp.hpp"

#include "intel_gpu/op/gated_mlp.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <cmath>
#include <numeric>

namespace ov::intel_gpu {

FuseFCSwiGLUToGatedMLP::FuseFCSwiGLUToGatedMLP() {
    using namespace ov::pass::pattern;

    auto src = any_input();
    auto w_up = any_input();
    auto w_gate = any_input();
    auto w_down = any_input();

    auto mm_up = wrap_type<ov::op::v0::MatMul>({src, w_up});
    auto mm_gate = wrap_type<ov::op::v0::MatMul>({src, w_gate});
    auto swish = wrap_type<ov::op::v4::Swish>({mm_gate});
    auto mul_0 = wrap_type<ov::op::v1::Multiply>({swish, mm_up});
    auto mul_1 = wrap_type<ov::op::v1::Multiply>({mm_up, swish});
    auto mul = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_0, mul_1});
    auto mm_down = wrap_type<ov::op::v0::MatMul>({mul, w_down});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto down = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(mm_down).get_node_shared_ptr());
        auto up = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(mm_up).get_node_shared_ptr());
        auto gate = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(mm_gate).get_node_shared_ptr());
        auto sw = ov::as_type_ptr<ov::op::v4::Swish>(pm.at(swish).get_node_shared_ptr());
        ov::NodeVector new_ops;

        if (!down || !up || !gate || !sw || transformation_callback(down))
            return false;

        // const auto is_mlp1 = [](const std::shared_ptr<ov::Node>& node) {
        //     if (!node)
        //         return false;
        //     const auto& name = node->get_friendly_name();
        //     return name.find("layers.1.mlp") != std::string::npos ||
        //            name.find("layers/1/mlp") != std::string::npos;
        // };

        // if (!is_mlp1(down) || !is_mlp1(up) || !is_mlp1(gate))
        //     return false;

        if (down->get_transpose_a() || up->get_transpose_a() || gate->get_transpose_a())
            return false;

        if (up->input_value(0) != gate->input_value(0))
            return false;

        // Swish can be represented as 1-input(default beta=1.0) or 2-input(beta tensor).
        if (sw->get_input_size() != 1 && sw->get_input_size() != 2)
            return false;

        if (sw->get_input_size() == 2) {
            auto beta_const = ov::as_type_ptr<ov::op::v0::Constant>(sw->get_input_node_shared_ptr(1));
            if (!beta_const)
                return false;
            auto beta = beta_const->cast_vector<float>();
            if (beta.empty() || std::fabs(beta[0] - 1.0f) > 1e-6f)
                return false;
        }

        auto create_transpose = [&](const ov::Output<ov::Node>& node, const std::string& transpose_name) -> ov::Output<ov::Node> {
            std::vector<size_t> transpose_order(node.get_partial_shape().size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{transpose_order.size()}, transpose_order);
            auto transpose = std::make_shared<ov::op::v1::Transpose>(node, transpose_const);
            if (!ov::is_type<ov::op::v0::Constant>(transpose)) {
                new_ops.push_back(transpose_const);
                MatcherPass::register_new_node(transpose);
            }
            transpose->set_friendly_name(transpose_name);
            ov::disable_constant_folding(transpose);
            new_ops.push_back(transpose);
            return transpose->output(0);
        };

        ov::Output<ov::Node> gate_weights = gate->input_value(1);
        ov::Output<ov::Node> up_weights = up->input_value(1);
        ov::Output<ov::Node> down_weights = down->input_value(1);
        if (gate->get_transpose_b()) {
            gate_weights = create_transpose(gate->input_value(1), gate->get_friendly_name() + "/transpose_b_for_gmlp");
        }
        if (up->get_transpose_b()) {
            up_weights = create_transpose(up->input_value(1), up->get_friendly_name() + "/transpose_b_for_gmlp");
        }
        if (down->get_transpose_b()) {
            down_weights = create_transpose(down->input_value(1), down->get_friendly_name() + "/transpose_b_for_gmlp");
        }

        struct decompression_info {
            bool compressed = false;
            bool has_zero_point = false;
            ov::Output<ov::Node> weight;
            ov::Output<ov::Node> scale;
            ov::Output<ov::Node> zero_point;
        };

        auto is_compressed_const = [](const ov::Output<ov::Node>& output) {
            const auto et = output.get_element_type();
            return et == ov::element::u8 || et == ov::element::i8 || et == ov::element::u4 || et == ov::element::i4;
        };

        auto parse_decompression = [&](const ov::Output<ov::Node>& maybe_decompressed_weight) -> decompression_info {
            decompression_info info;

            auto direct_convert = ov::as_type_ptr<ov::op::v0::Convert>(maybe_decompressed_weight.get_node_shared_ptr());
            if (direct_convert) {
                auto compressed_weights = direct_convert->input_value(0);
                if (is_compressed_const(compressed_weights)) {
                    auto scale_const = ov::op::v0::Constant::create(direct_convert->get_output_element_type(0), ov::Shape{1}, {1.0f});
                    new_ops.push_back(scale_const);
                    info.compressed = true;
                    info.has_zero_point = false;
                    info.weight = compressed_weights;
                    info.scale = scale_const;
                    return info;
                }
            }

            auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(maybe_decompressed_weight.get_node_shared_ptr());
            if (!mul) {
                return info;
            }

            ov::Output<ov::Node> dequant_input;
            ov::Output<ov::Node> scale_input;
            auto lhs_convert = ov::as_type_ptr<ov::op::v0::Convert>(mul->get_input_node_shared_ptr(0));
            auto lhs_subtract = ov::as_type_ptr<ov::op::v1::Subtract>(mul->get_input_node_shared_ptr(0));
            auto rhs_convert = ov::as_type_ptr<ov::op::v0::Convert>(mul->get_input_node_shared_ptr(1));
            auto rhs_subtract = ov::as_type_ptr<ov::op::v1::Subtract>(mul->get_input_node_shared_ptr(1));

            if (lhs_convert || lhs_subtract) {
                dequant_input = mul->input_value(0);
                scale_input = mul->input_value(1);
            } else if (rhs_convert || rhs_subtract) {
                dequant_input = mul->input_value(1);
                scale_input = mul->input_value(0);
            } else {
                return info;
            }

            auto convert = ov::as_type_ptr<ov::op::v0::Convert>(dequant_input.get_node_shared_ptr());
            if (convert) {
                auto compressed_weights = convert->input_value(0);
                if (!is_compressed_const(compressed_weights)) {
                    return info;
                }

                info.compressed = true;
                info.has_zero_point = false;
                info.weight = compressed_weights;
                info.scale = scale_input;
                return info;
            }

            auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(dequant_input.get_node_shared_ptr());
            if (!subtract) {
                return info;
            }

            ov::Output<ov::Node> convert_output;
            ov::Output<ov::Node> zp_output;

            auto lhs_convert_in_sub = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(0));
            auto rhs_convert_in_sub = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(1));

            if (lhs_convert_in_sub) {
                convert_output = subtract->input_value(0);
                zp_output = subtract->input_value(1);
            } else if (rhs_convert_in_sub) {
                convert_output = subtract->input_value(1);
                zp_output = subtract->input_value(0);
            } else {
                return info;
            }

            auto compressed_weights = convert_output.get_node_shared_ptr()->input_value(0);
            if (!is_compressed_const(compressed_weights)) {
                return info;
            }

            if (auto zp_convert = ov::as_type_ptr<ov::op::v0::Convert>(zp_output.get_node_shared_ptr())) {
                auto candidate_zp = zp_convert->input_value(0);
                if (ov::is_type<ov::op::v0::Constant>(candidate_zp.get_node())) {
                    zp_output = candidate_zp;
                }
            }

            const auto weight_et = compressed_weights.get_element_type();
            const auto zp_et = zp_output.get_element_type();
            if ((weight_et == ov::element::u8 || weight_et == ov::element::u4) && zp_et == ov::element::u4) {
                auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_output, ov::element::u8);
                new_ops.push_back(zp_convert);
                MatcherPass::register_new_node(zp_convert);
                zp_output = zp_convert;
            } else if ((weight_et == ov::element::i8 || weight_et == ov::element::i4) && zp_et == ov::element::i4) {
                auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_output, ov::element::i8);
                new_ops.push_back(zp_convert);
                MatcherPass::register_new_node(zp_convert);
                zp_output = zp_convert;
            }

            info.compressed = true;
            info.has_zero_point = true;
            info.weight = compressed_weights;
            info.scale = scale_input;
            info.zero_point = zp_output;
            return info;
        };

        std::shared_ptr<ov::Node> gmlp;
        auto gate_dq = parse_decompression(gate->input_value(1));
        auto up_dq = parse_decompression(up->input_value(1));
        auto down_dq = parse_decompression(down->input_value(1));

        const bool compressed_all = gate_dq.compressed && up_dq.compressed && down_dq.compressed;
        const bool has_zp_all = gate_dq.has_zero_point && up_dq.has_zero_point && down_dq.has_zero_point;
        const bool has_no_zp_all = !gate_dq.has_zero_point && !up_dq.has_zero_point && !down_dq.has_zero_point;

        if (compressed_all && (has_zp_all || has_no_zp_all)) {
            auto transpose_for_compressed = [&](ov::Output<ov::Node> tensor, bool need_transpose, const std::string& name_suffix) {
                if (!need_transpose) {
                    return tensor;
                }

                const auto rank = tensor.get_partial_shape().rank();
                if (!rank.is_static() || rank.get_length() < 2) {
                    return tensor;
                }

                return create_transpose(tensor, name_suffix);
            };

            auto gate_weight = transpose_for_compressed(gate_dq.weight,
                                                        gate->get_transpose_b(),
                                                        gate->get_friendly_name() + "/transpose_compressed_weight_for_gmlp");
            auto up_weight = transpose_for_compressed(up_dq.weight,
                                                      up->get_transpose_b(),
                                                      up->get_friendly_name() + "/transpose_compressed_weight_for_gmlp");
            auto down_weight = transpose_for_compressed(down_dq.weight,
                                                        down->get_transpose_b(),
                                                        down->get_friendly_name() + "/transpose_compressed_weight_for_gmlp");

            auto gate_scale = transpose_for_compressed(gate_dq.scale,
                                                       gate->get_transpose_b() && ov::shape_size(gate_dq.scale.get_shape()) > 1,
                                                       gate->get_friendly_name() + "/transpose_compressed_scale_for_gmlp");
            auto up_scale = transpose_for_compressed(up_dq.scale,
                                                     up->get_transpose_b() && ov::shape_size(up_dq.scale.get_shape()) > 1,
                                                     up->get_friendly_name() + "/transpose_compressed_scale_for_gmlp");
            auto down_scale = transpose_for_compressed(down_dq.scale,
                                                       down->get_transpose_b() && ov::shape_size(down_dq.scale.get_shape()) > 1,
                                                       down->get_friendly_name() + "/transpose_compressed_scale_for_gmlp");

            if (has_zp_all) {
                auto gate_zp = transpose_for_compressed(gate_dq.zero_point,
                                                        gate->get_transpose_b() && ov::shape_size(gate_dq.zero_point.get_shape()) > 1,
                                                        gate->get_friendly_name() + "/transpose_compressed_zp_for_gmlp");
                auto up_zp = transpose_for_compressed(up_dq.zero_point,
                                                      up->get_transpose_b() && ov::shape_size(up_dq.zero_point.get_shape()) > 1,
                                                      up->get_friendly_name() + "/transpose_compressed_zp_for_gmlp");
                auto down_zp = transpose_for_compressed(down_dq.zero_point,
                                                        down->get_transpose_b() && ov::shape_size(down_dq.zero_point.get_shape()) > 1,
                                                        down->get_friendly_name() + "/transpose_compressed_zp_for_gmlp");

                gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(up->input_value(0),
                                                                      gate_weight,
                                                                      up_weight,
                                                                      down_weight,
                                                                      gate_scale,
                                                                      up_scale,
                                                                      down_scale,
                                                                      gate_zp,
                                                                      up_zp,
                                                                      down_zp,
                                                                      ov::op::internal::GLU::GluType::Swish,
                                                                      down->get_output_element_type(0));
            } else {
                gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(up->input_value(0),
                                                                      gate_weight,
                                                                      up_weight,
                                                                      down_weight,
                                                                      gate_scale,
                                                                      up_scale,
                                                                      down_scale,
                                                                      ov::op::internal::GLU::GluType::Swish,
                                                                      down->get_output_element_type(0));
            }
        }

        if (!gmlp) {
            gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(up->input_value(0),
                                                                  gate_weights,
                                                                  up_weights,
                                                                  down_weights,
                                                                  ov::op::internal::GLU::GluType::Swish,
                                                                  down->get_output_element_type(0));
        }

        gmlp->set_friendly_name(down->get_friendly_name());
        new_ops.push_back(gmlp);
        ov::copy_runtime_info(m.get_matched_nodes(), new_ops);
        ov::replace_node(down, gmlp);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(mm_down, "FuseFCSwiGLUToGatedMLP");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
