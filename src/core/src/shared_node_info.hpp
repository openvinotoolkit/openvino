// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>

namespace ov {
class SharedRTInfo {
public:
    SharedRTInfo() : m_use_topological_cache(false), m_function_has_changed(false) {}

    void set_use_topological_cache(bool status) {
        m_use_topological_cache = status;
        m_function_has_changed = true;
    }

    void set_function_has_changed(bool status) {
        m_function_has_changed = status;
    }

    bool get_use_topological_cache() const {
        return m_use_topological_cache;
    }

    bool get_function_has_changed() const {
        return m_function_has_changed;
    }

private:
    bool m_use_topological_cache;
    bool m_function_has_changed;
};
}  // namespace ov
