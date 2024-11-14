// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "eltwise_to_eltwise_tpp.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "transformations/tpp/x64/op/factory.hpp"

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reduce.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

EltwiseToEltwiseTPP::EltwiseToEltwiseTPP() {
    MATCHER_SCOPE(EltwiseToEltwiseTPP);

    auto is_supported_by_tpp = [](const Output<Node>& out) {
        return op::NodeFactory::is_supported(out.get_node_shared_ptr());
    };
    auto supported_eltwise = ov::pass::pattern::wrap_type<ov::op::util::UnaryElementwiseArithmetic,
                                                          ov::op::util::BinaryElementwiseArithmetic,
                                                          ov::snippets::op::ReduceBase>(is_supported_by_tpp);


    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EltwiseToEltwiseTPP")
        const auto node = m.get_match_root();
        if (node->is_dynamic()) {
            return false;
        }

        const auto& tpp_eltwise = op::NodeFactory::create(node);
        OPENVINO_ASSERT(tpp_eltwise, "Failed to create TPP node");

        const size_t M_block = 32;
        const size_t N_block = ov::is_type<ov::snippets::op::ReduceBase>(node) ? ov::snippets::utils::get_full_dim_value() : 64;
        ov::replace_node_update_name(node, tpp_eltwise);
        for (size_t i = 0; i < node->get_input_size(); i++)
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(tpp_eltwise->input(i), {M_block, N_block});

        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(tpp_eltwise->output(0), {M_block, N_block});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(supported_eltwise, matcher_name);
    register_matcher(m, callback);
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
