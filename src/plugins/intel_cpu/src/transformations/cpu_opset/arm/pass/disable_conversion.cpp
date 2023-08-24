// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "disable_conversion.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

#include <openvino/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::DisableConversion::DisableConversion() {
    auto m_parent_convert = ngraph::pattern::any_input();
    auto m_convert = ngraph::pattern::wrap_type<ov::op::v0::Convert>({m_parent_convert});
    auto m_child_convert1 = ngraph::pattern::wrap_type<ov::op::v0::MatMul>({m_convert});
    auto m_child_convert2 = ngraph::pattern::wrap_type<ov::op::v0::MatMul>({m_convert});

    ngraph::matcher_pass_callback callback = [&](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto parent_convert = pattern_map.at(m_parent_convert);
        const auto convert = pattern_map.at(m_convert);

        const auto child_convert1 = pattern_map.at(m_child_convert1);
        const auto child_convert2 = pattern_map.at(m_child_convert2);
        auto matmul1 = std::dynamic_pointer_cast<ov::op::v0::MatMul>(child_convert1.get_node_shared_ptr());
        auto matmul2 = std::dynamic_pointer_cast<ov::op::v0::MatMul>(child_convert2.get_node_shared_ptr());

        auto convert1 = std::make_shared<ov::op::v0::Convert>(parent_convert, child_convert1.get_element_type());
        auto convert2 = std::make_shared<ov::op::v0::Convert>(parent_convert, child_convert2.get_element_type());
        copy_runtime_info(convert.get_node_shared_ptr(), convert1);
        copy_runtime_info(convert.get_node_shared_ptr(), convert2);

        auto new_matmul1 = matmul1->clone_with_new_inputs({convert1->output(0)});
        auto new_matmul2 = matmul2->clone_with_new_inputs({convert2->output(0)});
        copy_runtime_info(matmul1, new_matmul1);
        copy_runtime_info(matmul2, new_matmul2);

        ngraph::replace_node(m.get_match_root(), {convert1, convert2, new_matmul1, new_matmul2});
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(m_parent_convert, "DisableConversion");
    register_matcher(m, callback);
}
