// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util.hpp"

#include "../../logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace util {

namespace opp = ov::pass::pattern;

// TODO: visualize
ShapeOfToConst::ShapeOfToConst(const std::shared_ptr<ov::Model>& model) {
    auto key_val = opp::wrap_type<ov::op::v0::Parameter>();
    auto shapeof = opp::wrap_type<ov::op::v3::ShapeOf>({key_val});
    auto inds = opp::wrap_type<ov::op::v0::Constant>();
    auto axis = opp::wrap_type<ov::op::v0::Constant>();
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shapeof, inds, axis});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_node_key_val = node_to_output.at(key_val).get_node_shared_ptr();
        auto matched_node_shapeof = node_to_output.at(shapeof).get_node_shared_ptr();
        auto matched_node_gather = node_to_output.at(gather).get_node_shared_ptr();

        auto gather_in_node = matched_node_gather->input(0).get_node();
        auto param_in_tensor = matched_node_shapeof->input(0).get_tensor_ptr();

        // Only looking for Params introduced from stateful model
        if (param_in_tensor->get_shape().size() != 4) {
            return false;  // root hasn't changed
        }

        // Create a new static Const instead of Param->ShapeOf
        auto new_const = std::make_shared<ov::op::v0::Constant>(gather_in_node->get_element_type(),
                                                                ov::Shape{4},
                                                                param_in_tensor->get_shape().data());

        // Drop connection from Param to ShapeOf
        for (auto&& node_reader_port : matched_node_key_val->output(0).get_target_inputs()) {
            if (node_reader_port.get_node() == matched_node_shapeof.get()) {
                matched_node_key_val->output(0).remove_target_input(node_reader_port);
                break;
            }
        }

        // Drop connection from ShapeOf to Gather
        for (auto&& node_outputs : matched_node_shapeof->outputs()) {
            for (auto&& node_reader_port : node_outputs.get_target_inputs()) {
                if (node_reader_port.get_node() == matched_node_gather.get()) {
                    node_outputs.remove_target_input(node_reader_port);
                    break;
                }
            }
        }

        // Reconnect old gather reader to the new Const
        matched_node_gather->input(0).replace_source_output(new_const);

        // Drop Parameter if ShapeOf was the only reader
        NPUW_ASSERT(ov::op::util::is_parameter(matched_node_key_val));
        if (matched_node_key_val->output(0).get_target_inputs().size() == 0) {
            model->remove_parameter(std::dynamic_pointer_cast<ov::op::v0::Parameter>(matched_node_key_val));
        }

        return true;  // root has changed
    };
    register_matcher(std::make_shared<opp::Matcher>(gather, "TagShapeOfToConst"), std::move(callback));
}

}  // namespace util
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
