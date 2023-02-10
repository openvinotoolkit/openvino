// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/snippets_isa.hpp"

#include "snippets/utils.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
const std::set<std::vector<int>> FuseTransposeBrgemm::supported_cases = {{0, 2, 1, 3}};
FuseTransposeBrgemm::FuseTransposeBrgemm() {
    MATCHER_SCOPE(FuseTransposeBrgemm);
    auto transpose_is_supported = [](const Output<Node>& transpose_port) {
        const auto transpose_node = transpose_port.get_node_shared_ptr();
        // it's safe to do so because of the patterns we used. alternatively we can do it through pattern_values_map
        const auto& constant = as_type_ptr<ngraph::opset1::Constant>(transpose_node->get_input_node_shared_ptr(1));
        // if Transpose in and out layout is not empty => something was already fused on this port
        if (!utils::get_node_output_layout(transpose_node).empty() ||
            !utils::get_node_output_layout(transpose_node->get_input_node_shared_ptr(0)).empty())
            return false;
        const auto& transpose_order = constant->cast_vector<int>();
        // todo: this limitation is due to the fact that offsets are calculated in Kernel, and the only way
        //  to calc them non-default way is to set Parameter rt_info field. This limitation can be removed if
        //  the rt_info is properly propagated to the corresponding parameter
        if (!is_type<ngraph::opset1::Parameter>(transpose_node->get_input_node_shared_ptr(0)) ||
            supported_cases.count(transpose_order) == 0)
            return false;
        return true;
    };
    auto constant = pattern::wrap_type<opset1::Constant>();
    auto transpose = pattern::wrap_type<opset1::Transpose>({pattern::any_input(), constant}, transpose_is_supported);
    auto transpose_matcher = std::make_shared<pattern::Matcher>(transpose);
    auto brgemm_any = pattern::wrap_type<op::Brgemm>({pattern::any_input(), pattern::any_input()});

    auto brgemm_in0 = pattern::wrap_type<op::Brgemm>({transpose, pattern::any_input()});
    auto brgemm_in1 = pattern::wrap_type<op::Brgemm>({pattern::any_input(), transpose});
    auto brgemm_out0 = pattern::wrap_type<opset1::Transpose>({brgemm_any, constant});
    auto brgemm_or_transpose = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{brgemm_in0, brgemm_in1, brgemm_out0});

    auto callback = [=](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "FuseTransposeBrgemm")
        auto set_layout_from_order = [](const std::shared_ptr<opset1::Transpose>& node, const ov::Output<Node>& port) {
            const auto& const_order = as_type_ptr<opset1::Constant>(node->get_input_node_shared_ptr(1));
            std::vector<size_t> layout = const_order->cast_vector<size_t>();
            auto& rt_info = port.get_node_shared_ptr()->get_rt_info();
            rt_info["Layout"] = layout;
        };
        auto brgemm = as_type_ptr<op::Brgemm>(m.get_match_root());
        // Transpose on the Brgemm's output
        if (!brgemm) {
            brgemm = as_type_ptr<op::Brgemm>(m.get_match_root()->get_input_node_shared_ptr(0));
            const auto& brgemm_out = brgemm->output(0);
            const auto& transpose_out = m.get_match_value();
            for (const auto& in : transpose_out.get_target_inputs())
                in.replace_source_output(brgemm->output(0));
            set_layout_from_order(as_type_ptr<opset1::Transpose>(transpose_out.get_node_shared_ptr()), brgemm_out);
        }
        for (size_t i = 0; i < brgemm->get_input_size(); i++) {
            const auto& in_value = brgemm->input_value(i);
            if (transpose_matcher->match(in_value)) {
                const auto& transpose = as_type_ptr<opset1::Transpose>(in_value.get_node_shared_ptr());
                set_layout_from_order(transpose, transpose->input_value(0));
                brgemm->set_argument(i, transpose->input_value(0));
            }
        }
        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        brgemm->validate_and_infer_types();
        return true;
    };
    register_matcher(std::make_shared<pattern::Matcher>(brgemm_or_transpose, matcher_name), callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph