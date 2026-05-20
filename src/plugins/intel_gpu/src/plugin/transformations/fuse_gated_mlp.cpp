// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_gated_mlp.hpp"

#include "intel_gpu/op/gated_mlp.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <cmath>
#include <numeric>

namespace ov::intel_gpu {

FuseGatedMLP::FuseGatedMLP() {
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
            new_ops.push_back(transpose);
            return transpose->output(0);
        };

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

        auto extract_compression_info = [&](const ov::Output<ov::Node>& maybe_decompressed_weight) -> decompression_info {
            decompression_info info;
            info.weight = maybe_decompressed_weight;

            // Look through Reshape: grouped quantization stores weights as 3D [OC, ngroups, group_size]
            // with a Reshape to 2D [OC, IC] before MatMul.
            auto input_to_parse = maybe_decompressed_weight;
            std::shared_ptr<ov::op::v1::Reshape> top_reshape;
            if (ov::as_type_ptr<ov::op::v1::Reshape>(maybe_decompressed_weight.get_node_shared_ptr())) {
                top_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(maybe_decompressed_weight.get_node_shared_ptr());
                input_to_parse = top_reshape->input_value(0);
            }

            auto direct_convert = ov::as_type_ptr<ov::op::v0::Convert>(input_to_parse.get_node_shared_ptr());
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

            auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(input_to_parse.get_node_shared_ptr());
            if (!mul)
                return info;

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
        auto gate_info = extract_compression_info(gate->input_value(1));
        auto up_info = extract_compression_info(up->input_value(1));
        auto down_info = extract_compression_info(down->input_value(1));

        // Grouped quantization stores weights/scales/zp as 3D (e.g. [OC, ngroups, group_size]).
        // Flatten to 2D for GatedMLP/oneDNN.
        auto flatten_to_2d = [&](ov::Output<ov::Node>& tensor) {
            const auto& ps = tensor.get_partial_shape();
            if (!ps.rank().is_static() || ps.rank().get_length() <= 2)
                return;
            auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {0, -1});
            auto reshape = std::make_shared<ov::op::v1::Reshape>(tensor, target_shape, true);
            new_ops.push_back(target_shape);
            new_ops.push_back(reshape);
            tensor = reshape->output(0);
        };

        for (auto* info : {&gate_info, &up_info, &down_info}) {
            if (!info->compressed)
                continue;
            flatten_to_2d(info->weight);
            flatten_to_2d(info->scale);
            if (info->has_zero_point)
                flatten_to_2d(info->zero_point);
        }

        const bool compressed_all = gate_info.compressed && up_info.compressed && down_info.compressed;
        const bool has_zp_all = gate_info.has_zero_point && up_info.has_zero_point && down_info.has_zero_point;
        const bool has_no_zp_all = !gate_info.has_zero_point && !up_info.has_zero_point && !down_info.has_zero_point;

        // oneDNN gated_mlp expects weights in [OC, IC] physical layout. With ba format tag,
        // layout_to_memory_desc converts [OC, IC] to oneDNN logical dims [IC, OC] as expected
        // by the gated_mlp primitive descriptor.
        // When transpose_b=true, weights are already [OC, IC] — no transpose needed.
        // When transpose_b=false, weights are [IC, OC] and must be transposed to [OC, IC].
        if (!gate->get_transpose_b())
            gate_info.weight = create_transpose(gate_info.weight, gate->get_friendly_name() + "/transpose_gate");
        if (!up->get_transpose_b())
            up_info.weight = create_transpose(up_info.weight, up->get_friendly_name() + "/transpose_up");
        if (!down->get_transpose_b())
            down_info.weight = create_transpose(down_info.weight, down->get_friendly_name() + "/transpose_down");

        if (compressed_all && (has_zp_all || has_no_zp_all)) {
            if (has_zp_all) {
                gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(up->input_value(0),
                                                                      gate_info.weight,
                                                                      up_info.weight,
                                                                      down_info.weight,
                                                                      gate_info.scale,
                                                                      up_info.scale,
                                                                      down_info.scale,
                                                                      gate_info.zero_point,
                                                                      up_info.zero_point,
                                                                      down_info.zero_point,
                                                                      ov::op::internal::GLU::GluType::Swish,
                                                                      down->get_output_element_type(0));
            } else {
                gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(up->input_value(0),
                                                                      gate_info.weight,
                                                                      up_info.weight,
                                                                      down_info.weight,
                                                                      gate_info.scale,
                                                                      up_info.scale,
                                                                      down_info.scale,
                                                                      ov::op::internal::GLU::GluType::Swish,
                                                                      down->get_output_element_type(0));
            }
        }

        if (!gmlp) {
            gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(up->input_value(0),
                                                                  gate_info.weight,
                                                                  up_info.weight,
                                                                  down_info.weight,
                                                                  ov::op::internal::GLU::GluType::Swish,
                                                                  down->get_output_element_type(0));
        }

        gmlp->set_friendly_name(down->get_friendly_name());
        new_ops.push_back(gmlp);
        ov::copy_runtime_info(m.get_matched_nodes(), new_ops);
        ov::replace_node(down, gmlp);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(mm_down, "FuseGatedMLP");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
