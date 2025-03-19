// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/bias_attribute.hpp"
#include "low_precision/network_helper.hpp"

#include <iterator>
#include <memory>
#include "openvino/opsets/opset1.hpp"
#include <string>
#include <unordered_map>
#include <vector>

void ov::mark_as_bias(const std::shared_ptr<ov::Node>& node) {
    auto& rt = node->get_rt_info();
    rt[ov::BiasAttribute::get_type_info_static()] = ov::BiasAttribute();
}

bool ov::marked_as_bias(const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ov::BiasAttribute::get_type_info_static()) != rt_info.end();
}

bool ov::BiasAttribute::is_copyable(const std::shared_ptr<ov::Node>& to) const {
    return ov::is_type<ov::opset1::Add>(to) && ov::pass::low_precision::NetworkHelper::getConstantInput(to) != nullptr;
}
