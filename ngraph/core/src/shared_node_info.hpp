// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/variant.hpp>

namespace ov {
class SharedRTInfo {
public:
    SharedRTInfo() {
        set_use_topological_cache(false);
    }

    const RTMap& get_rt_info() const {
        return m_rt_info;
    }

    RTMap& get_rt_info() {
        return m_rt_info;
    }

    void set_use_topological_cache(bool status) {
        auto& info = get_rt_info();
        info["use_topological_cache"] = std::make_shared<VariantWrapper<int64_t>>(status);
    }

    bool get_use_topological_cache() const {
        const auto& info = get_rt_info();
        if (info.count("use_topological_cache") == 0) {
            throw Exception("use_topological_cache is not set");
        }
        return std::dynamic_pointer_cast<VariantWrapper<int64_t>>(info.at("use_topological_cache"))->get();
    }

private:
    RTMap m_rt_info;
};
}  // namespace ov
