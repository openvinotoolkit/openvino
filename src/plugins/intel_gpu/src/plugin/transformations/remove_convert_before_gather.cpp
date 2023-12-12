// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_convert_before_gather.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace intel_gpu {

RemoveConvertBeforeGather::RemoveConvertBeforeGather() {
    using namespace ov::pass::pattern;

    // Detect Any input - Convert - Gather pattern
    auto weights = wrap_type<ov::op::v0::Constant>();
    auto indices = any_input();
    auto axis = any_input();
    auto convert = wrap_type<ov::op::v0::Convert>({weights});
    auto gather = wrap_type<ov::op::v8::Gather>({convert, indices, axis});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto weights_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights).get_node_shared_ptr());
        const auto& convert_node = pattern_map.at(convert).get_node_shared_ptr();
        const auto& gather_node = pattern_map.at(gather).get_node_shared_ptr();

        if (convert_node->get_output_target_inputs(0).size() != 1)
            return true;

        std::cout << "RemoveConvertBeforeGather @@ " << convert_node->get_friendly_name() << std::endl;

        auto& compressed_type = convert_node->get_input_element_type(0);

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

        // ov::replace_node(convert_node, input_node);
        // if (is_decompression(convert_node)) {
        //     unmark_as_decompression(convert_node);
        //     disable_constant_folding(convert_node);
        // }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather);
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
