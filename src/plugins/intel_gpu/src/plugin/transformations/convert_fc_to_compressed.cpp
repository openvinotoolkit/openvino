// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fc_to_compressed.hpp"

#include <memory>

#include "compressed_weights_pattern.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

ConvertFullyConnectedToFullyConnectedCompressed::ConvertFullyConnectedToFullyConnectedCompressed() {
    auto data_m = any_input();
    auto bias_m = any_input();

    FC_COMPRESSED_WEIGHT_PATTERN

    auto fully_connected_m = wrap_type<op::FullyConnected>({data_m, compressed_weights_input_m, bias_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected_m));
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(decompressed_weights_m));
        OPENVINO_ASSERT(pattern_map.count(bias_m));
        auto fc = ov::as_type_ptr<op::FullyConnected>(pattern_map.at(fully_connected_m).get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }
        const bool has_transpose_before_reshape = pattern_map.count(transpose_before_reshape_m) != 0u;
        const bool has_transpose = pattern_map.count(transpose_after_reshape_m) != 0u || has_transpose_before_reshape;
        auto scale_shape = pattern_map.at(mul_const_m).get_shape();
        bool sub_with_convert = pattern_map.count(sub_with_convert_m) > 0;

        auto weight_shape = fc->get_input_shape(1);
        const auto output_features_idx =
            fc->get_transpose_b() ? weight_shape.size() - 2 : weight_shape.size() - 1;
        const auto output_features = weight_shape[output_features_idx];
        auto has_output_features_in_inner_dimension = [output_features](const std::shared_ptr<ov::Node>& node) {
            const auto& shape = node->get_output_shape(0);
            const auto inner_dim = std::find_if(shape.rbegin(), shape.rend(), [](size_t dim) {
                return dim > 1;
            });
            return inner_dim != shape.rend() && *inner_dim == output_features;
        };
        bool is_weight_3d = (std::count_if(weight_shape.begin(), weight_shape.end(), [](size_t d) {
                                 return d > 1;
                             }) == 3);
        bool grouped = scale_shape.size() == weight_shape.size() + 1;

        bool weight_u8 = false;
        std::shared_ptr<ov::Node> weight_ptr =
            pattern_map.count(weights_const_m) ? pattern_map.at(weights_const_m).get_node_shared_ptr() : pattern_map.at(weights_param_m).get_node_shared_ptr();
        if (weight_ptr->get_element_type() == ov::element::u8 || weight_ptr->get_element_type() == ov::element::i8) {
            weight_u8 = true;
        }

        auto reshape_const = [has_transpose, grouped, is_weight_3d](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant != nullptr);
            ov::Shape current_shape = constant->get_shape();
            if (current_shape.size() <= 2) {
                return constant;
            }

            ov::Shape new_shape;
            if (current_shape.size() == 3) {
                if (is_weight_3d) {
                    return constant;
                }
                new_shape = (has_transpose || !grouped) ? ov::Shape{current_shape[0] * current_shape[1], current_shape[2]}
                                                        : ov::Shape{current_shape[0], current_shape[1] * current_shape[2]};
            } else if (current_shape.size() == 4 && is_weight_3d) {
                new_shape = (has_transpose || !grouped) ? ov::Shape{current_shape[0], current_shape[1] * current_shape[2], current_shape[3]}
                                                        : ov::Shape{current_shape[0], current_shape[1], current_shape[2] * current_shape[3]};
            } else if (current_shape.size() == 4 && !is_weight_3d) {
                new_shape = (has_transpose || !grouped) ? ov::Shape{current_shape[0] * current_shape[1] * current_shape[2], current_shape[3]}
                                                        : ov::Shape{current_shape[0] * current_shape[1], current_shape[2] * current_shape[3]};
            } else {
                OPENVINO_THROW("Unexpected constant shape rank ", current_shape.size(), " with is_weight_3d=", is_weight_3d);
            }
            auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);

            ov::copy_weightless_cache_attr(constant, new_constant);
            return new_constant;
        };

        auto convert_const_to_u8 = [&](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            std::shared_ptr<ov::Node> result = nullptr;
            // Convert ZP to u8
            if (constant->get_element_type() == ov::element::u8) {
                result = constant;
            } else if (constant->get_element_type() == ov::element::u4 || constant->get_element_type() == ov::element::u2) {
                result = std::make_shared<ov::op::v0::Convert>(node, ov::element::u8);
                // Only unsigned ZP types can be converted to u8.
            } else if (weight_u8 && sub_with_convert && !constant->get_element_type().is_signed()) {
                result = std::make_shared<ov::op::v0::Convert>(node, ov::element::u8);
            } else {
                result = constant;
            }

            ov::copy_weightless_cache_attr(node, result);
            return result;
        };

        const ov::Output<Node>& fc_input_a = fc->input(0).get_source_output();
        const auto& scale = has_transpose_before_reshape
                                ? pattern_map.at(mul_const_m).get_node_shared_ptr()
                                : reshape_const(pattern_map.at(mul_const_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> optional_zero_point = nullptr;

        const bool with_zero_point = pattern_map.count(sub_no_convert_m) > 0 || pattern_map.count(sub_with_convert_m) > 0;
        if (with_zero_point) {
            const auto zero_point = has_transpose_before_reshape
                                        ? pattern_map.at(sub_const_m).get_node_shared_ptr()
                                        : reshape_const(pattern_map.at(sub_const_m).get_node_shared_ptr());
            optional_zero_point = convert_const_to_u8(zero_point);
        }

        std::shared_ptr<ov::Node> fc_input_b =
            pattern_map.count(weights_const_m)
                ? (has_transpose_before_reshape ? pattern_map.at(weights_const_m).get_node_shared_ptr()
                                                : reshape_const(pattern_map.at(weights_const_m).get_node_shared_ptr()))
                : (pattern_map.count(weights_reshape_m) ? pattern_map.at(weights_reshape_m).get_node_shared_ptr()
                                                        : pattern_map.at(weights_param_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> fc_input_scale = scale;
        std::shared_ptr<ov::Node> fc_input_zp = optional_zero_point;
        std::shared_ptr<ov::Node> fc_input_bias = pattern_map.at(bias_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};

        if (fc_input_b->get_output_partial_shape(0).size() != fc_input_scale->get_shape().size()) {
            OPENVINO_ASSERT(!pattern_map.count(weights_const_m));
            ov::Shape weight_shape_final(fc_input_scale->get_shape().size(), 1);
            for (size_t i = weight_shape.size() - 1, idx = fc_input_scale->get_shape().size() - 1;; --i) {
                if (weight_shape[i] > 1) {
                    weight_shape_final[idx--] = weight_shape[i];
                }
                if (i == 0) {
                    break;
                }
            }
            if (has_transpose) {
                std::swap(weight_shape_final[0], weight_shape_final[1]);
            }
            std::shared_ptr<ov::Node> weight_shape_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{weight_shape_final.size()}, weight_shape_final);
            fc_input_b = std::make_shared<ov::op::v1::Reshape>(fc_input_b, weight_shape_const, false);
            result_nodes.push_back(weight_shape_const);
            result_nodes.push_back(fc_input_b);
        }

        if (has_transpose) {
            const auto& transpose = pattern_map
                                        .at(has_transpose_before_reshape ? transpose_before_reshape_input_m : transpose_after_reshape_m)
                                        .get_node_shared_ptr();
            std::shared_ptr<ov::Node> transpose_const = has_transpose_before_reshape
                                                            ? transpose->get_input_node_shared_ptr(1)
                                                            : pattern_map.at(transpose_const_m).get_node_shared_ptr();
            if (ov::shape_size(transpose_const->get_shape()) != fc_input_b->get_output_partial_shape(0).size()) {
                std::vector<int32_t> new_order(fc_input_b->get_output_partial_shape(0).size());
                std::iota(new_order.begin(), new_order.end(), 0);
                std::swap(new_order[new_order.size() - 1], new_order[new_order.size() - 2]);
                transpose_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{new_order.size()}, new_order);
            }

            fc_input_b = transpose->clone_with_new_inputs({fc_input_b->output(0), transpose_const});
            result_nodes.push_back(fc_input_b);

            if (ov::shape_size(scale->output(0).get_shape()) > 1 &&
                !has_output_features_in_inner_dimension(scale)) {
                fc_input_scale = transpose->clone_with_new_inputs({scale->output(0), transpose_const});
                result_nodes.push_back(fc_input_scale);
            }

            if (with_zero_point && ov::shape_size(optional_zero_point->output(0).get_shape()) > 1 &&
                !has_output_features_in_inner_dimension(optional_zero_point)) {
                fc_input_zp = transpose->clone_with_new_inputs({optional_zero_point->output(0), transpose_const});
                result_nodes.push_back(fc_input_zp);
            }
        }

        if (has_transpose_before_reshape) {
            const auto& reshape = pattern_map.at(transpose_before_reshape_m).get_node_shared_ptr();
            const auto& reshape_const_node = pattern_map.at(transpose_const_m);
            fc_input_b = reshape->clone_with_new_inputs({fc_input_b->output(0), reshape_const_node});
            result_nodes.push_back(fc_input_b);

            auto reshape_squeeze = [&](std::shared_ptr<ov::Node> node) {
                const auto& shape = node->get_output_shape(0);
                const auto output_rank = fc_input_b->get_output_shape(0).size();
                if (shape.size() <= output_rank) {
                    return node;
                }

                ov::Shape output_shape(shape.begin(), shape.begin() + output_rank - 1);
                output_shape.push_back(ov::shape_size(ov::Shape(shape.begin() + output_rank - 1, shape.end())));
                auto shape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                ov::Shape{output_shape.size()},
                                                                output_shape);
                auto reshaped = std::make_shared<ov::op::v1::Reshape>(node, shape_const, false);
                result_nodes.push_back(shape_const);
                result_nodes.push_back(reshaped);
                return std::static_pointer_cast<ov::Node>(reshaped);
            };

            fc_input_scale = reshape_squeeze(fc_input_scale);
            if (with_zero_point) {
                fc_input_zp = reshape_squeeze(fc_input_zp);
            }
        }

        if (pattern_map.count(mul2_m)) {
            auto mul2_op_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(mul2_const_m).get_node_shared_ptr());
            fc_input_scale = ov::op::util::make_try_fold<ov::op::v1::Multiply>(fc_input_scale, mul2_op_const);
        }

        std::shared_ptr<ov::Node> new_fc = nullptr;
        if (with_zero_point) {
            new_fc = std::make_shared<op::FullyConnectedCompressed>(fc_input_a,
                                                                    fc_input_b,
                                                                    fc_input_bias,
                                                                    fc_input_scale,
                                                                    fc_input_zp,
                                                                    fc->get_output_type(),
                                                                    fc->get_transpose_b());
        } else {
            new_fc = std::make_shared<op::FullyConnectedCompressed>(fc_input_a,
                                                                    fc_input_b,
                                                                    fc_input_bias,
                                                                    fc_input_scale,
                                                                    fc->get_output_type(),
                                                                    fc->get_transpose_b());
        }

        result_nodes.push_back(new_fc);
        new_fc->set_friendly_name(fc->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(fc, new_fc);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "ConvertFullyConnectedToFullyConnectedCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
