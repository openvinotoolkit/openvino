// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "eltwise_to_eltwise_tpp.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "transformations/snippets/tpp/op/factory.hpp"

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/util/arithmetic_reduction.hpp"
#include "snippets/lowered/port_descriptor.hpp"

#include "snippets/op/reduce.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {
// todo: this is copied from brgemm_to_brgemm_cpu. move to common utils.
namespace {
using namespace snippets::lowered;
template<typename T>
void set_port_desc(const T& port, std::vector<size_t> subtensor) {
    const auto& shape = port.get_shape();
    for (size_t i = 1; i <= std::min(subtensor.size(), shape.size()); i++) {
        auto& dim = subtensor[subtensor.size() - i];
        if (dim != PortDescriptor::ServiceDimensions::FULL_DIM)
            dim = std::min(dim, shape[shape.size() - i]);
    }
    PortDescriptorUtils::set_port_descriptor_ptr(port, std::make_shared<PortDescriptor>(shape, subtensor));
}
} // namespace

EltwiseToEltwiseTPP::EltwiseToEltwiseTPP() {
    MATCHER_SCOPE(EltwiseToEltwiseTPP);

    auto is_supported_by_tpp = [](const Output<Node>& out) {
        return op::TPPNodeFactory::is_supported(out.get_node_shared_ptr());
    };
    // const auto& unary = ov::pass::pattern::wrap_type<ov::op::util::UnaryElementwiseArithmetic>(is_supported_by_tpp);
    // const auto& binary = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>(is_supported_by_tpp);
    // auto supported_eltwise = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{unary, binary});
    auto supported_eltwise = ov::pass::pattern::wrap_type<ov::op::util::UnaryElementwiseArithmetic,
                                                          ov::op::util::BinaryElementwiseArithmetic,
                                                          ov::snippets::op::ReduceBase>(is_supported_by_tpp);


    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EltwiseToEltwiseTPP")
        const auto node = m.get_match_root();
        if (node->is_dynamic()) {
            return false;
        }

        const auto& tpp_eltwise = op::TPPNodeFactory::create(node);
        OPENVINO_ASSERT(tpp_eltwise, "Failed to create TPP node");

        const size_t M_block = 32;
        const size_t N_block = ov::is_type<ov::snippets::op::ReduceBase>(node) ?
                               PortDescriptor::ServiceDimensions::FULL_DIM :
                               64;
        ngraph::replace_node(node, tpp_eltwise);
        for (size_t i = 0; i < node->get_input_size(); i++)
            set_port_desc(tpp_eltwise->input(i), {M_block, N_block});

        set_port_desc(tpp_eltwise->output(0), {M_block, N_block});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(supported_eltwise, matcher_name);
    register_matcher(m, callback);
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
