// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>

namespace ov {
class SharedRTInfo {
public:
    SharedRTInfo() : m_use_topological_cache(false) {}

    void set_use_topological_cache(bool status) {
        m_use_topological_cache = status;
    }

    bool get_use_topological_cache() const {
        return m_use_topological_cache;
    }

private:
    bool m_use_topological_cache;
};
}  // namespace ov
