// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/transpose_decomposition.hpp"

#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace snippets {
namespace pass {
using namespace lowered;

const std::set<std::vector<int>> TransposeDecomposition::supported_cases = {{0, 2, 3, 1}};

TransposeDecomposition::TransposeDecomposition() {
    MATCHER_SCOPE(TransposeDecomposition);
    // Todo: we need a special transformation that detects and propagates data access pattern to Parameters and Results
    //       this is needed to communicate access pattern to the plugin node and op::Kernel
    //       This is the reason we match only to Parameter, this limitation could be relaxed if we propagate access pattern
    //       to the appropriate parameter
    auto match_data = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto match_order = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto match_transpose = ov::pass::pattern::wrap_type<ov::opset1::Transpose>({match_data, match_order});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TransposeDecomposition")
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto& data_input = pattern_to_output.at(match_data);
        const auto transpose = ov::as_type_ptr<ov::opset1::Transpose>(pattern_to_output.at(match_transpose).get_node_shared_ptr());

        const auto order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(match_order).get_node_shared_ptr());
        if (transformation_callback(transpose) || transpose->is_dynamic())
            return false;

        auto order_value = order->cast_vector<int>();
        if (supported_cases.count(order_value) == 0)
            return false;

        // number of elements that can be processed on every iteration. For 0,1,2,3 -> 0,2,3,1 we can guarantee only scalar access
        const auto subtensor = std::vector<size_t>{1};
        const auto& layout = order->cast_vector<size_t>();

        // todo: LoadReshape used here is essentially Load + an easy way to maintain correct shape propagation
        //  fix this in future and develop a more consistent shape propagation approach.
        auto load = std::make_shared<snippets::op::LoadReshape>(data_input, subtensor[0], 0, layout);
        auto store = std::make_shared<snippets::op::Store>(load, subtensor[0]);

        PortDescriptorUtils::set_port_descriptor_ptr(load->input(0),
                                                     std::make_shared<PortDescriptor>(load->get_input_partial_shape(0).to_shape(), subtensor, layout));
        PortDescriptorUtils::set_port_descriptor_ptr(load->output(0),
                                                     std::make_shared<PortDescriptor>(load->get_output_partial_shape(0).to_shape(), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(store->input(0),
                                                     std::make_shared<PortDescriptor>(store->get_input_partial_shape(0).to_shape(), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(store->output(0),
                                                     std::make_shared<PortDescriptor>(store->get_output_partial_shape(0).to_shape(), subtensor));

        for (auto& input : transpose->output(0).get_target_inputs()) {
            input.replace_source_output(store->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(match_transpose, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
