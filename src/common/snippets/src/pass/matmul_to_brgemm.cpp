// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/matmul_to_brgemm.hpp"

#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {

using namespace lowered;

MatMulToBrgemm::MatMulToBrgemm() {
    MATCHER_SCOPE(MatMulToBrgemm);
    auto matmul_pattern = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});

    auto callback = [matmul_pattern](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MatMulToBrgemm")
        const auto& pm = m.get_pattern_value_map();
        const auto matmul = as_type_ptr<ov::opset1::MatMul>(pm.at(matmul_pattern).get_node_shared_ptr());
        // Brgemm doesn't support transposed inputs currently, so we don't convert such matmuls
        if (matmul->get_transpose_a())
            return false;

        auto generate_layout = [](const ov::PartialShape& shape, const bool transpose) {
            std::vector<size_t> layout(shape.size());
            std::iota(layout.begin(), layout.end(), 0);
            if (transpose)
                std::swap(*layout.rbegin(), *(layout.rbegin() + 1));
            return layout;
        };

        const auto layout_a = generate_layout(matmul->get_input_partial_shape(0), matmul->get_transpose_a());
        const auto layout_b = generate_layout(matmul->get_input_partial_shape(1), matmul->get_transpose_b());
        const auto brgemm = std::make_shared<op::Brgemm>(matmul->input_value(0), matmul->input_value(1), 0, 0, 0, layout_a, layout_b);

        static const std::vector<size_t> subtensor{utils::get_full_dim_value(), utils::get_full_dim_value()};
        PortDescriptorUtils::set_port_descriptor(brgemm->input(0), subtensor, layout_a);
        PortDescriptorUtils::set_port_descriptor(brgemm->input(1), subtensor, layout_b);
        PortDescriptorUtils::set_port_descriptor(brgemm->output(0), subtensor);

        ov::NodeVector nodes = { brgemm };
        if (brgemm->get_output_element_type(0) != matmul->get_output_element_type(0)) {
            nodes.emplace_back(std::make_shared<op::ConvertSaturation>(brgemm, matmul->get_output_element_type(0)));
        }
        brgemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(matmul, nodes);
        ov::replace_node(matmul, nodes.back());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
