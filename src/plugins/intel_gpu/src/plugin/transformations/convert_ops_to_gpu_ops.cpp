// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_ops_to_gpu_ops.hpp"

#include <memory>

#include "intel_gpu/op/fully_connected.hpp"
#include <ov_ops/fully_connected.hpp>
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

ConvertExtensionOp::ConvertExtensionOp() {
    add_matcher<ConvertFullyConnectedCommonToFullyConnected>();
}

ConvertFullyConnectedCommonToFullyConnected::ConvertFullyConnectedCommonToFullyConnected() {
    auto data_m = any_input();
    auto bias_m = any_input();
    auto weights_m = any_input();

    auto fully_connected_m = wrap_type<ov::op::internal::FullyConnected>({data_m, weights_m, bias_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(fully_connected_m));
        OPENVINO_ASSERT(pattern_map.count(weights_m));
        OPENVINO_ASSERT(pattern_map.count(bias_m));
        auto fc = ov::as_type_ptr<ov::op::internal::FullyConnected>(pattern_map.at(fully_connected_m).get_node_shared_ptr());
        if (!fc || transformation_callback(fc)) {
            return false;
        }


        const ov::Output<Node>& fc_input_a = fc->input(0).get_source_output();
        std::shared_ptr<ov::Node> fc_input_weight = pattern_map.at(weights_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> fc_input_bias = pattern_map.at(bias_m).get_node_shared_ptr();
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};

        std::shared_ptr<ov::Node> new_fc = std::make_shared<op::FullyConnected>(fc_input_a,
                                                                                fc_input_weight,
                                                                                fc_input_bias,
                                                                                fc->get_output_type());

        result_nodes.push_back(new_fc);
        new_fc->set_friendly_name(fc->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(fc, new_fc);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, "ConvertFullyConnectedCommonToFullyConnected");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
