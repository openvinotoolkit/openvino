// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_fc_dimensions.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ReduceFCDimensions::ReduceFCDimensions() {
    auto activations_m = ov::pass::pattern::any_input(ov::pass::pattern::shape_matches("[1, 1, ?, ?]"));
    auto weights_m = ov::pass::pattern::any_input(ov::pass::pattern::shape_matches("[?, ?]"));
    auto no_bias_m = ov::pass::pattern::wrap_type<op::Placeholder>();
    auto fc_m = ov::pass::pattern::wrap_type<op::FullyConnected>({activations_m, weights_m, no_bias_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto activations = pattern_map.at(activations_m).get_node_shared_ptr();
        auto weights = pattern_map.at(weights_m).get_node_shared_ptr();
        auto no_bias = pattern_map.at(no_bias_m).get_node_shared_ptr();
        auto fc = pattern_map.at(fc_m).get_node_shared_ptr();
       
        auto wei_pshape = weights->get_output_partial_shape(0);
        // Do not apply in case of dynamic weight shape
        if (wei_pshape.is_dynamic()) {
            return false;
        }
        auto squeeze_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, -1, wei_pshape[1].get_length()});
        auto squeeze = std::make_shared<ov::op::v1::Reshape>(activations, squeeze_const, false);
        ov::copy_runtime_info(activations, squeeze);
        squeeze->set_friendly_name(activations->get_friendly_name() + "_squeeze");

        auto fc_new = fc->clone_with_new_inputs({squeeze, weights, no_bias});
        ov::copy_runtime_info(fc, fc_new);

        auto unsqueeze_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, -1, wei_pshape[0].get_length()});
        ov::copy_runtime_info(fc, unsqueeze_const);
        auto unsqueeze = std::make_shared<ov::op::v1::Reshape>(fc_new, unsqueeze_const, false);
        unsqueeze->set_friendly_name(fc->get_friendly_name() + "_unsqueeze");
        ov::copy_runtime_info(fc, unsqueeze);

        ov::replace_node(fc, unsqueeze);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, "ReduceFCDimensions");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
