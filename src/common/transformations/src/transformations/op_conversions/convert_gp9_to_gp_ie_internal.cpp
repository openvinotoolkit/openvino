// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gp9_to_gp_ie_internal.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/generate_proposals.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/generate_proposals_ie_internal.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertGP9ToGPIEInternal::ConvertGP9ToGPIEInternal() {
    matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto root = m.get_match_root();
        const auto old_node = ov::as_type_ptr<ov::op::v9::GenerateProposals>(root);
        if (!old_node) {
            return false;
        }

        for (const auto& i : old_node->inputs()) {
            if (i.get_partial_shape().is_dynamic()) {
                return false;
            }
        }

        NodeVector new_ops;

        auto new_node = std::make_shared<op::internal::GenerateProposalsIEInternal>(old_node->input_value(0),
                                                                                    old_node->input_value(1),
                                                                                    old_node->input_value(2),
                                                                                    old_node->input_value(3),
                                                                                    old_node->get_attrs(),
                                                                                    old_node->get_roi_num_type());

        new_ops.push_back(new_node);
        Output<ov::Node> output_0 = new_node->output(0);
        new_ops.emplace_back(output_0.get_node_shared_ptr());
        Output<ov::Node> output_1 = new_node->output(1);
        new_ops.emplace_back(output_1.get_node_shared_ptr());
        Output<ov::Node> output_2 = new_node->output(2);
        new_ops.emplace_back(output_2.get_node_shared_ptr());

        new_node->set_friendly_name(old_node->get_friendly_name());
        copy_runtime_info(old_node, new_ops);
        replace_node(old_node, {output_0, output_1, output_2});
        return true;
    };

    const auto generate_proposals = ov::pass::pattern::wrap_type<ov::op::v9::GenerateProposals>();
    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(generate_proposals, "ConvertGP9ToGPIEInternal");
    register_matcher(matcher, callback);
}
