// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"

bool ov::WeightlessCacheAttribute::is_copyable() const {
    return false;
}

OPENVINO_API void ov::copy_weightless_cache_attr(const std::shared_ptr<ov::Node>& from,
                                                 const std::shared_ptr<ov::Node>& to) {
    const auto& rt_info = from->get_rt_info();
    auto weightless_caching_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());

    if (weightless_caching_attr != rt_info.end()) {
        to->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = weightless_caching_attr->second;
    }
}