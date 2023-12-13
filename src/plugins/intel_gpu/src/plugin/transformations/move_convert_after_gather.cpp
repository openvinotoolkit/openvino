// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_convert_after_gather.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace intel_gpu {

MoveConvertAfterGather::MoveConvertAfterGather() {
    using namespace ov::pass::pattern;

    // f16 compressed LLM word embedding pattern
    // Const -(f16)-> Convert -(f32)-> Gather -(f32)->
    // Convert is moved after Gather to avoid constant folding, which would double the memory usage.
    // Const -(f16)-> Gather -(f16)-> Convert -(f32)->
    auto weights = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto convert = wrap_type<ov::op::v0::Convert>({weights}, consumers_count(1));
    auto gather = wrap_type<ov::op::v8::Gather>({convert, any_input(), any_input()});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto weights_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights).get_node_shared_ptr());
        const auto& convert_node = pattern_map.at(convert).get_node_shared_ptr();
        const auto& gather_node = pattern_map.at(gather).get_node_shared_ptr();
        auto& compressed_type = convert_node->get_input_element_type(0);

        if (compressed_type != ov::element::f16)
            return false;

        weights_node->clear_control_dependents();
        gather_node->input(0).replace_source_output(weights_node->output(0));

        convert_node->clear_control_dependents();
        const auto& gather_target_inputs = gather_node->get_output_target_inputs(0);
        for (const auto& target_input : gather_target_inputs) {
            target_input.replace_source_output(convert_node->output(0));
        }

        gather_node->clear_control_dependents();
        gather_node->set_output_type(0, compressed_type, gather_node->get_output_partial_shape(0));
        convert_node->input(0).replace_source_output(gather_node->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather);
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
