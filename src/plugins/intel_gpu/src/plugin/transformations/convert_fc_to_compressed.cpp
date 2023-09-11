// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fc_to_compressed.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/op/subtract.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

ConvertFullyConnectedToFullyConnectedCompressed::ConvertFullyConnectedToFullyConnectedCompressed() {
    using namespace ov::pass::pattern;

    auto weights_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m});

    auto sub_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto subtract_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});

    auto mul_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m});
    auto mul_no_sub_m = wrap_type<ov::op::v1::Multiply>({convert_m, mul_const_m});
    auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_with_sub_m, mul_no_sub_m});

    auto transpose_const_m = wrap_type<ov::op::v0::Constant>();
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({mul_m, transpose_const_m});
    auto weights_input_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{mul_m, transpose_m});

    auto data_m = any_input();
    auto fully_connected_m = wrap_type<op::FullyConnected>({data_m, weights_input_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected_m));
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(weights_m));
        OPENVINO_ASSERT(pattern_map.count(convert_m));
        auto fc = std::dynamic_pointer_cast<op::FullyConnected>(pattern_map.at(fully_connected_m).get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        const auto& fc_input_a = fc->get_input_node_shared_ptr(0);
        const auto& scale = pattern_map.at(mul_const_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> optional_zero_point = nullptr;

        ov::NodeVector nodes_to_copy_info{pattern_map.at(fully_connected_m).get_node_shared_ptr(),
                                          pattern_map.at(convert_m).get_node_shared_ptr()};
        if (pattern_map.count(mul_no_sub_m)) {
            nodes_to_copy_info.push_back(pattern_map.at(mul_no_sub_m).get_node_shared_ptr());
        }
        if (pattern_map.count(mul_with_sub_m)) {
            nodes_to_copy_info.push_back(pattern_map.at(mul_with_sub_m).get_node_shared_ptr());
        }

        const bool with_zero_point = pattern_map.count(subtract_m) > 0;
        if (with_zero_point) {
            optional_zero_point = pattern_map.at(sub_const_m).get_node_shared_ptr();
            nodes_to_copy_info.push_back(subtract_m);
        }

        std::shared_ptr<ov::Node> fc_input_b = pattern_map.at(weights_m).get_node_shared_ptr();
        if (pattern_map.count(transpose_m)) {
            const auto& transpose = pattern_map.at(transpose_m).get_node_shared_ptr();
            const auto& transpose_const = pattern_map.at(transpose_const_m).get_node_shared_ptr();
            fc_input_b = transpose->clone_with_new_inputs({ fc_input_b->output(0), transpose_const });
        }

        std::shared_ptr<ov::Node> new_fc = nullptr;
        if (with_zero_point) {
            new_fc = std::make_shared<op::FullyConnectedCompressed>(fc_input_a,
                                                                    fc_input_b,
                                                                    scale,
                                                                    optional_zero_point,
                                                                    fc->get_output_type());
        } else {
            new_fc = std::make_shared<op::FullyConnectedCompressed>(fc_input_a,
                                                                    fc_input_b,
                                                                    scale,
                                                                    fc->get_output_type());
        }

        new_fc->set_friendly_name(fc->get_friendly_name());
        ov::copy_runtime_info(nodes_to_copy_info, new_fc);
        ov::replace_node(fc, new_fc);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m);
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
