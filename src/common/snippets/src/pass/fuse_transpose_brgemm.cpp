// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/snippets_isa.hpp"

#include "snippets/utils.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

namespace ov {
namespace snippets {
namespace pass {

const std::set<std::vector<int>> FuseTransposeBrgemm::supported_cases = {{0, 2, 1, 3}};

bool FuseTransposeBrgemm::is_supported_transpose(const Output<Node>& transpose_port) {
    const auto transpose_node = transpose_port.get_node_shared_ptr();
    // it's safe to do so because of the patterns we used. alternatively we can do it through pattern_values_map
    const auto& constant = as_type_ptr<ov::opset1::Constant>(transpose_node->get_input_node_shared_ptr(1));
    // if Transpose in and out layout is not empty => something was already fused on this port
    auto default_layout = std::vector<size_t>(transpose_port.get_partial_shape().size());
    std::iota(default_layout.begin(), default_layout.end(), 0);// NCHW layout by default
    if (lowered::PortDescriptorUtils::get_port_descriptor_ptr(transpose_port)->get_layout() != default_layout ||
        lowered::PortDescriptorUtils::get_port_descriptor_ptr(transpose_node->input_value(0))->get_layout() != default_layout)
        return false;
    const auto& transpose_order = constant->cast_vector<int>();
    // todo: this limitation is due to the fact that offsets are calculated in Kernel, and the only way
    //  to calc them non-default way is to set Parameter rt_info field. This limitation can be removed if
    //  the rt_info is properly propagated to the corresponding parameter
    return is_type<ov::opset1::Parameter>(transpose_node->get_input_node_shared_ptr(0)) &&
           supported_cases.count(transpose_order) != 0;
}

FuseTransposeBrgemm::FuseTransposeBrgemm() {
    MATCHER_SCOPE(FuseTransposeBrgemm);
    auto constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({ov::pass::pattern::any_input(), constant}, is_supported_transpose);
    auto transpose_matcher = std::make_shared<ov::pass::pattern::Matcher>(transpose);

    // Pattern 0: Transpose on 0-th input of MatMul
    auto brgemm_in0 = ov::pass::pattern::wrap_type<op::Brgemm>({transpose, ov::pass::pattern::any_input()});

    // Pattern 1: Transpose on 1-st input of MatMul
    auto brgemm_in1 = ov::pass::pattern::wrap_type<op::Brgemm>({ov::pass::pattern::any_input(), transpose});

    // Pattern 2: Transpose on output of MatMul
    auto brgemm_out = ov::pass::pattern::wrap_type<op::Brgemm>({ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});
    auto transpose2 = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({brgemm_out, constant});

    auto brgemm_or_transpose = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{brgemm_in0, brgemm_in1, transpose2});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "FuseTransposeBrgemm")
        auto brgemm = ov::as_type_ptr<op::Brgemm>(m.get_match_root());

        // Transpose on the Brgemm's output
        if (!brgemm) {
            brgemm = ov::as_type_ptr<op::Brgemm>(m.get_match_root()->get_input_node_shared_ptr(0));
            const auto& brgemm_out = brgemm->output(0);
            const auto& transpose_out = m.get_match_value();
            const auto& const_order = ov::as_type_ptr<ov::op::v0::Constant>(transpose_out.get_node_shared_ptr()->get_input_node_shared_ptr(1));
            const auto& original_port = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(brgemm_out);
            original_port->set_shape(transpose_out.get_partial_shape().to_shape());
            original_port->set_layout(const_order->cast_vector<size_t>());
            for (const auto& in : transpose_out.get_target_inputs())
                in.replace_source_output(brgemm->output(0));
        }

        for (size_t i = 0; i < brgemm->get_input_size(); i++) {
            const auto& in = brgemm->input(i);
            const auto& in_value = in.get_source_output();
            if (transpose_matcher->match(in_value)) {
                const auto& transpose = as_type_ptr<ov::op::v1::Transpose>(in_value.get_node_shared_ptr());
                const auto& const_order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
                brgemm->set_argument(i, transpose->input_value(0));
                const auto& original_port = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(in);
                original_port->set_shape(transpose->get_input_partial_shape(0).to_shape());
                original_port->set_layout(const_order->cast_vector<size_t>());
            }
        }

        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        brgemm->validate_and_infer_types();
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(brgemm_or_transpose, matcher_name), callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
