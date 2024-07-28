// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "scalar_to_scalar_tpp.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/op/scalar.hpp"
#include "transformations/tpp/x64/op/scalar.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"
#include "snippets/lowered/port_connector.hpp"


namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

ScalarToScalarTPP::ScalarToScalarTPP() {
    MATCHER_SCOPE(ScalarToScalarTPP);

    auto snippets_scalar = ov::pass::pattern::wrap_type<ov::snippets::op::Scalar>();


    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::ScalarToScalarTPP")
        const auto node = ov::as_type_ptr<ov::snippets::op::Scalar>(m.get_match_root());
        OPENVINO_ASSERT(node, "Failed to obtain a valid Scalar Op in ScalarToScalarTPP");
        size_t num_connected_tpp = 0;
        const auto& target_ins = node->get_output_target_inputs(0);
        for (const auto& in : target_ins) {
            if (dynamic_cast<tpp::modifier::TensorProcessingPrimitive*>(in.get_node()))
                num_connected_tpp++;
        }
        if (num_connected_tpp == 0)
            return false;
        // Note: If needed, we can support cases when scalar has TPP and non-TPP consumers if we copy the scalar.
        // However, this is rarely needed in practice and the assert is here to flag invalid configurations.
        OPENVINO_ASSERT(num_connected_tpp == target_ins.size(), "Either all or none Scalar outputs should be TPP");

        const auto& tpp_scalar = std::make_shared<tpp::op::Scalar>(*node);
        tpp_scalar->set_friendly_name(node->get_friendly_name());
        ov::replace_node_update_name(node, tpp_scalar);
        const auto& out = tpp_scalar->output(0);
        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(out, {1});
        for (const auto& in : out.get_target_inputs())
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(in, {1});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(snippets_scalar, matcher_name);
    register_matcher(m, callback);
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
