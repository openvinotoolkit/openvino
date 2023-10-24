// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/matmul_to_brgemm.hpp"

#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/port_descriptor.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace snippets {
namespace pass {

void MatMulToBrgemm::init_ports(const std::shared_ptr<op::Brgemm>& brgemm) const {
    auto get_subtensor = [](const ov::Shape& shape) {
        return std::vector<size_t>{ lowered::PortDescriptor::ServiceDimensions::FULL_DIM, lowered::PortDescriptor::ServiceDimensions::FULL_DIM };
    };
    for (const auto& input : brgemm->inputs()) {
        const auto tensor = input.get_partial_shape().to_shape();
        const auto subtensor = get_subtensor(tensor);
        lowered::PortDescriptorUtils::set_port_descriptor_ptr(input, std::make_shared<lowered::PortDescriptor>(tensor, subtensor));
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto tensor = brgemm->get_output_shape(0);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto subtensor = get_subtensor(tensor);
    lowered::PortDescriptorUtils::set_port_descriptor_ptr(brgemm->output(0), std::make_shared<lowered::PortDescriptor>(tensor, subtensor));
}

MatMulToBrgemm::MatMulToBrgemm() {
    MATCHER_SCOPE(MatMulToBrgemm);
    auto matmul_pattern = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MatMulToBrgemm")
        auto& pm = m.get_pattern_value_map();
        const auto matmul = as_type_ptr<ov::opset1::MatMul>(pm.at(matmul_pattern).get_node_shared_ptr());
        // Brgemm doesn't support transposed inputs currently, so we don't convert such matmuls
        if (matmul->get_transpose_a() || matmul->get_transpose_b())
            return false;

        auto brgemm = std::make_shared<op::Brgemm>(matmul->get_input_source_output(0), matmul->get_input_source_output(1));
        ov::NodeVector nodes = { brgemm };
        if (brgemm->get_output_element_type(0) != matmul->get_output_element_type(0)) {
            nodes.emplace_back(std::make_shared<op::ConvertSaturation>(brgemm, matmul->get_output_element_type(0)));
        }
        brgemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(matmul, nodes);
        ov::replace_node(matmul, nodes.back());
        init_ports(brgemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
