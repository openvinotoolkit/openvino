// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/nonconvertible_divide.hpp"

void ov::disable_divide_conversion(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[NonconvertibleDivide::get_type_info_static()] = NonconvertibleDivide{};
}

void ov::enable_divide_conversion(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(NonconvertibleDivide::get_type_info_static());
}

bool ov::divide_is_nonconvertible(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(NonconvertibleDivide::get_type_info_static());
}
