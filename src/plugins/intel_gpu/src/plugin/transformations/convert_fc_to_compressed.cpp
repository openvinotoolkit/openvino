// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fc_to_compressed.hpp"

#include <memory>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
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
#include "openvino/core/graph_util.hpp"

#include "compressed_weights_pattern.hpp"

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
        OPENVINO_ASSERT(pattern_map.count(compressed_weights_m));
        OPENVINO_ASSERT(pattern_map.count(bias_m));
        OPENVINO_ASSERT(pattern_map.count(convert_m));
        auto fc = ov::as_type_ptr<op::FullyConnected>(pattern_map.at(fully_connected_m).get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }
        bool has_transpose = pattern_map.count(transpose_m);
        auto scale_shape = pattern_map.at(mul_const_m).get_shape();
        bool grouped = std::count_if(scale_shape.begin(), scale_shape.end(), [](size_t d) { return d > 1; }) > 1;
        bool sub_with_convert = (pattern_map.count(sub_with_convert_m) > 0) ? true : false;

        auto weight_shape = fc->get_input_shape(1);
        bool is_weight_3d = (std::count_if(weight_shape.begin(), weight_shape.end(), [](size_t d) { return d > 1; }) == 3);

        auto weight_ptr = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(compressed_weights_m).get_node_shared_ptr());
        bool weight_u8 = false;
        if (weight_ptr->get_element_type() == ov::element::u8 || weight_ptr->get_element_type() == ov::element::i8)
            weight_u8 = true;

        auto reshape_const = [has_transpose, grouped, is_weight_3d](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant != nullptr);
            ov::Shape current_shape = constant->get_shape();
            if (current_shape.size() <= 2)
                return constant;

            ov::Shape new_shape;
            if (current_shape.size() == 3) {
                if (is_weight_3d)
                    return constant;
                else
                    new_shape = (has_transpose || !grouped) ? ov::Shape{current_shape[0] * current_shape[1], current_shape[2]}
                                                            : ov::Shape{current_shape[0], current_shape[1] * current_shape[2]};
            } else {
                OPENVINO_ASSERT(current_shape.size() == 4 && is_weight_3d);
                    new_shape = (has_transpose || !grouped) ? ov::Shape{current_shape[0], current_shape[1] * current_shape[2], current_shape[3]}
                                                            : ov::Shape{current_shape[0], current_shape[1], current_shape[3] * current_shape[1]};
            }
            auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);

            ov::copy_weightless_cache_attr(constant, new_constant);
            return new_constant;
        };

        auto convert_const_to_u8 = [&](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            std::shared_ptr<ov::Node> result = nullptr;
            // Convert ZP to u8
            if (constant->get_element_type() == ov::element::u8)
                result = std::dynamic_pointer_cast<ov::Node>(constant);
            else if (constant->get_element_type() == ov::element::u4)
                result = std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(node, ov::element::u8));
            else if (weight_u8 && sub_with_convert)
                result = std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(node, ov::element::u8));
            else
                result = std::dynamic_pointer_cast<ov::Node>(constant);

            ov::copy_weightless_cache_attr(node, result);
            return result;
        };


        const ov::Output<Node>& fc_input_a = fc->input(0).get_source_output();
        const auto& scale = reshape_const(pattern_map.at(mul_const_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> optional_zero_point = nullptr;

        const bool with_zero_point = pattern_map.count(sub_no_convert_m) > 0 || pattern_map.count(sub_with_convert_m) > 0;
        if (with_zero_point) {
            optional_zero_point = convert_const_to_u8(reshape_const(pattern_map.at(sub_const_m).get_node_shared_ptr()));
        }

        std::shared_ptr<ov::Node> fc_input_b = reshape_const(pattern_map.at(compressed_weights_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> fc_input_scale = scale;
        std::shared_ptr<ov::Node> fc_input_zp = optional_zero_point;
        std::shared_ptr<ov::Node> fc_input_bias = pattern_map.at(bias_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};

        if (has_transpose) {
            const auto& transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
            std::shared_ptr<ov::Node> transpose_const = pattern_map.at(transpose_const_m).get_node_shared_ptr();
            if (ov::shape_size(transpose_const->get_shape()) != fc_input_b->get_output_partial_shape(0).size()) {
                std::vector<int32_t> new_order(fc_input_b->get_output_partial_shape(0).size());
                std::iota(new_order.begin(), new_order.end(), 0);
                std::swap(new_order[new_order.size() - 1], new_order[new_order.size() - 2]);
                transpose_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{new_order.size()}, new_order);
            }

            fc_input_b = transpose->clone_with_new_inputs({ fc_input_b->output(0), transpose_const });
            result_nodes.push_back(fc_input_b);

            if (ov::shape_size(scale->output(0).get_shape()) > 1) {
                fc_input_scale = transpose->clone_with_new_inputs({ scale->output(0), transpose_const });
                result_nodes.push_back(fc_input_scale);
            }

            if (with_zero_point && ov::shape_size(optional_zero_point->output(0).get_shape()) > 1) {
                fc_input_zp = transpose->clone_with_new_inputs({ optional_zero_point->output(0), transpose_const });
                result_nodes.push_back(fc_input_zp);
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
                                                                    fc->get_output_type());
        } else {
            new_fc = std::make_shared<op::FullyConnectedCompressed>(fc_input_a,
                                                                    fc_input_b,
                                                                    fc_input_bias,
                                                                    fc_input_scale,
                                                                    fc->get_output_type());
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
