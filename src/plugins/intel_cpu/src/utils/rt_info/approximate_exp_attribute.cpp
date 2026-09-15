// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "approximate_exp_attribute.hpp"

#include <memory>

#include "openvino/core/node.hpp"

namespace ov::intel_cpu {

ApproximateExp::~ApproximateExp() = default;

void mark_as_approximate_exp(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[ApproximateExp::get_type_info_static()] = ApproximateExp{};
}

bool is_approximate_exp(const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ApproximateExp::get_type_info_static()) != rt_info.end();
}

}  // namespace ov::intel_cpu
