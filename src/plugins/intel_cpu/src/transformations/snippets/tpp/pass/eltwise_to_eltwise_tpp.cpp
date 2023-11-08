// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "eltwise_to_eltwise_tpp.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "transformations/snippets/tpp/op/eltwise.hpp"

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"


namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

EltwiseToEltwiseTPP::EltwiseToEltwiseTPP() {
    MATCHER_SCOPE(EltwiseToEltwiseTPP);

    auto is_supported_by_tpp = [](const Output<Node>& out) {
        return op::BinaryEltwiseTPP::is_supported(out.get_node_shared_ptr());
    };
    auto eltwise_label = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>(is_supported_by_tpp);

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EltwiseToEltwiseTPP")
        const auto node = m.get_match_root();
        if (node->is_dynamic()) {
            return false;
        }

        // todo: create TPP op factory
        std::shared_ptr<ov::op::Op> tpp_eltwise = nullptr;
        const auto& in0 = node->get_input_source_output(0);
        const auto& in1 = node->get_input_source_output(1);
        const auto& autob = node->get_autob();
        if (ov::is_type<ov::op::v1::Add>(node)) {
            tpp_eltwise = std::make_shared<tpp::op::Add>(in0, in1, autob);
        } else if (ov::is_type<ov::op::v1::Subtract>(node)) {
            tpp_eltwise = std::make_shared<tpp::op::Subtract>(in0, in1, autob);
        } else if (ov::is_type<ov::op::v1::Multiply>(node)) {
            tpp_eltwise = std::make_shared<tpp::op::Multiply>(in0, in1, autob);
        } else if (ov::is_type<ov::op::v1::Divide>(node)) {
            tpp_eltwise = std::make_shared<tpp::op::Divide>(in0, in1, autob);
        } else {
            OPENVINO_THROW("Unsupported ov::op for TPP conversion. Check the matcher");
        }

        tpp_eltwise->set_friendly_name(node->get_friendly_name());
        ngraph::replace_node(node, tpp_eltwise);

        // Transfer ports
//        set_port_desc(brgemm_cpu->input(0), brgemm_in0_desc->get_shape(), brgemm_in0_desc->get_subtensor(), brgemm_in0_desc->get_layout());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(eltwise_label, matcher_name);
    register_matcher(m, callback);
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
