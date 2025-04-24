// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/while.hpp"

#include <algorithm>

using namespace std;
using namespace ov;

op::internal::While::While(const OutputVector& inputs,
                           int32_t sub_block,
                           const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos)
    : Op(inputs),
      m_sub_block(sub_block),
      m_output_infos(output_infos) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::While::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<While>(new_args, m_sub_block, m_output_infos);
}

bool op::internal::While::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("sub_block", m_sub_block);
    return true;
}

void op::internal::While::validate_and_infer_types() {
    for (size_t i = 0; i < m_output_infos.size(); i++) {
        set_output_type(i, m_output_infos[i].first, m_output_infos[i].second);
    }
}
