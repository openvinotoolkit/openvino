// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/shutdown.hpp"

void shutdown_resources();

#include <functional>
#include <vector>

namespace {

class ShutdownRegistry {
public:
    std::vector<std::function<void()>> _callbacks;

    static ShutdownRegistry& get() {
        static ShutdownRegistry instance;
        return instance;
    }

    bool register_callback(const std::function<void()>& func) {
        _callbacks.emplace_back(func);
        return true;
    }

    ~ShutdownRegistry() {
        for (auto& func : _callbacks) {
            if (func) {
                func();
            }
        }
        _callbacks.clear();
    }
};
}

namespace ov {

bool register_shutdown_callback(const std::function<void()>& func) {
    return ShutdownRegistry::get().register_callback(func);
}

}  // namespace ov
