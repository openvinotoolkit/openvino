// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/shutdown.hpp"

#include <functional>
#include <vector>

namespace ov {

class ShutdownRegistry {
public:
    std::vector<std::function<void()>> _callbacks;

    static ShutdownRegistry& get() {
        static ShutdownRegistry instance;
        return instance;
    }

    bool register_callback(const std::function<void()>& func) {
        if (!func) {
            return false;
        }
        _callbacks.emplace_back(func);
        return true;
    }

    ~ShutdownRegistry() {
        for (auto& func : _callbacks) {
            func();
        }
        _callbacks.clear();
    }
};

bool register_shutdown_callback(const std::function<void()>& func) {
    return ShutdownRegistry::get().register_callback(func);
}

}  // namespace ov
