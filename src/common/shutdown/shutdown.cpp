// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/shutdown.hpp"

#include <functional>
#include <vector>

namespace ov {

class ShutdownRegistry {
private:
    std::vector<std::function<void()>> m_callbacks;
    ShutdownRegistry() = default;
public:

    static ShutdownRegistry& get() {
        static ShutdownRegistry instance;
        return instance;
    }

    bool register_callback(const std::function<void()>& func) {
        if (!func) {
            return false;
        }
        m_callbacks.emplace_back(func);
        return true;
    }

    ~ShutdownRegistry() {
        for (auto& func : m_callbacks) {
            func();
        }
        m_callbacks.clear();
    }
};

bool register_shutdown_callback(const std::function<void()>& func) {
    return ShutdownRegistry::get().register_callback(func);
}

}  // namespace ov
