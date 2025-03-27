// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/rt_info/external_parameter.hpp"

namespace ov::snippets {

void mark_as_external_parameter(const std::shared_ptr<Node>& node) {
    auto& rt = node->get_rt_info();
    rt[ExternalParameterAttribute::get_type_info_static()] = ExternalParameterAttribute();
}

bool is_external_parameter(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ExternalParameterAttribute::get_type_info_static()) != rt_info.end();
}

}  // namespace ov::snippets
