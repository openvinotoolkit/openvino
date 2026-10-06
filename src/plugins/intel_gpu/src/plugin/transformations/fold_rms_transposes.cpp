// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fold_rms_transposes.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rms.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

FoldRMSTransposes::FoldRMSTransposes() {
    auto input_m = any_input();
    auto input_order_m = wrap_type<ov::op::v0::Constant>();
    auto input_transpose_m = wrap_type<ov::op::v1::Transpose>({input_m, input_order_m}, consumers_count(1));
    auto gamma_m = any_input();
    auto rms_m = wrap_type<ov::op::internal::RMS>({input_transpose_m, gamma_m}, consumers_count(1));
    auto output_order_m = wrap_type<ov::op::v0::Constant>();
    auto output_transpose_m = wrap_type<ov::op::v1::Transpose>({rms_m, output_order_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        if (transformation_callback(matcher.get_match_root())) {
            return false;
        }

        const auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
        if (!rms || !rms->get_elementwise_affine() || rms->get_axis() != -1) {
            return false;
        }

        const auto input_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(input_order_m).get_node_shared_ptr())->cast_vector<int64_t>();
        const auto output_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(output_order_m).get_node_shared_ptr())->cast_vector<int64_t>();
        if (input_order.size() != output_order.size() || input_order.size() < 2) {
            return false;
        }

        for (size_t i = 0; i < input_order.size(); ++i) {
            if (input_order[i] < 0 || static_cast<size_t>(input_order[i]) >= input_order.size() || output_order[input_order[i]] != static_cast<int64_t>(i)) {
                return false;
            }
        }

        const auto axis = input_order.back();
        if (axis != 1 || (input_order.size() != 4 && input_order.size() != 5)) {
            return false;
        }

        const auto input = pattern_map.at(input_m);
        const auto gamma = pattern_map.at(gamma_m);
        auto new_rms = std::make_shared<ov::op::internal::RMS>(input, gamma, rms->get_epsilon(), rms->get_output_element_type(0), axis);
        new_rms->set_friendly_name(matcher.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(matcher.get_matched_nodes(), new_rms);
        ov::replace_node(matcher.get_match_root(), new_rms);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(output_transpose_m, "FoldRMSTransposes");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
