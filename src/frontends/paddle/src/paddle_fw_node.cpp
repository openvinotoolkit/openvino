// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <paddle_fw_node.hpp>

namespace ov {
namespace frontend {
namespace paddle {
void FrameworkNode::validate_and_infer_types() {
    ov::op::util::FrameworkNode::validate_and_infer_types();
    size_t idx = 0;
    for (const auto& port_pair : m_decoder->get_output_type_map()) {
        for (const auto& p_type : port_pair.second) {
            set_output_type(idx++, p_type, PartialShape::dynamic());
        }
    }
}

std::map<std::string, OutputVector> FrameworkNode::get_named_inputs() const {
    return m_decoder->map_for_each_input([&](const std::string& name, size_t) {
        auto it = std::find(m_inputs_names.begin(), m_inputs_names.end(), name);
        if (it != m_inputs_names.end()) {
            return input(it - m_inputs_names.begin()).get_source_output();
        } else {
            return Output<Node>();
        }
    });
}

std::map<std::string, OutputVector> FrameworkNode::return_named_outputs() {
    return m_decoder->map_for_each_output([&](const std::string&, size_t idx) {
        return output(idx);
    });
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
