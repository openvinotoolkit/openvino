// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_precision_change_flag.hpp"

void ov::intel_gna::rt_info::add_precision_change_flag(ov::Input<Node>& node,
    const ov::element::Type& in, const ov::element::Type& out) {
    RTMap& rt_info = node.get_rt_info();
    rt_info[GNAPrecisionChangeFlag::get_type_info_static()] = GNAPrecisionChangeFlag{in, out};
}

void ov::intel_gna::rt_info::remove_precision_change_flag(ov::Input<Node>& node) {
    RTMap& rt_info = node.get_rt_info();
    auto it = rt_info.find(GNAPrecisionChangeFlag::get_type_info_static());
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

bool ov::intel_gna::rt_info::is_precision_changed(const ov::Input<Node>& node) {
    const RTMap& rt_info = node.get_rt_info();
    if (rt_info.count(GNAPrecisionChangeFlag::get_type_info_static()) > 0) {
        auto flag = rt_info.at(GNAPrecisionChangeFlag::get_type_info_static()).as<GNAPrecisionChangeFlag>();
        return flag.is_changed();
    }
    return false;
}
