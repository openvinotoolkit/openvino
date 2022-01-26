// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/precision_sensitive_attribute.hpp"

void ov::mark_as_precision_sensitive(ov::Input<ov::Node> node_input) {
    auto& rt_info = node_input.get_rt_info();
    rt_info[PrecisionSensitive::get_type_info_static()] = PrecisionSensitive{};
}

void ov::unmark_as_precision_sensitive(ov::Input<ov::Node> node_input) {
    auto& rt_info = node_input.get_rt_info();
    rt_info.erase(PrecisionSensitive::get_type_info_static());
}

bool ov::is_precision_sensitive(const ov::Input<ov::Node>& node_input) {
    const auto& rt_info = node_input.get_rt_info();
    return rt_info.count(PrecisionSensitive::get_type_info_static());
}
