// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/merge_subsequent_dsr_operations.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

namespace vpu {

MergeSubsequentDSROperations::MergeSubsequentDSROperations() {
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        const auto& dsr = std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(m.get_match_root());
        if (!dsr) {
            return false;
        }

        const auto& predecessor = std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(dsr->input_value(0).get_node_shared_ptr());
        if (!predecessor) {
            return false;
        }
        // this will create a new DSR with correct inputs
        auto newDsr = dsr->clone_with_new_inputs({predecessor->input_value(0), dsr->input_value(1)});
        newDsr->set_friendly_name(dsr->get_friendly_name());
        // replace DSR2 with new so DSR2 will lose all consumers so it will die after pass execution
        ngraph::replace_node(dsr, newDsr);
        // reconnect all DSR1 consumers even with DSR2 which will be destructed so this is no more an issue
        for (auto &consumer : predecessor->get_output_target_inputs(0)) {
            consumer.replace_source_output(newDsr);
        }
        return true;
    };

    const auto& label = std::make_shared<ngraph::pattern::op::Label>(
        ngraph::element::i64,
        ngraph::Shape{},
        ngraph::pattern::has_class<ngraph::vpu::op::DynamicShapeResolver>());

    const auto& matcher = std::make_shared<ngraph::pattern::Matcher>(label, "MergeSubsequentDSROperations");
    register_matcher(matcher, callback);
}

}  // namespace vpu
