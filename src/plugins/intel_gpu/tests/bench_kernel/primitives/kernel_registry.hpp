// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <functional>
#include <memory>
#include <iostream>

#include "kernel_base.hpp"

namespace bench_kernel {

// ============================================================================
// Registry of all available kernel benchmarks
// ============================================================================

class kernel_registry {
public:
    using factory_fn = std::function<kernel_ptr()>;

    static kernel_registry& instance() {
        static kernel_registry reg;
        return reg;
    }

    void register_kernel(const std::string& name, factory_fn factory) {
        factories_[name] = std::move(factory);
    }

    kernel_ptr create(const std::string& name) const {
        auto it = factories_.find(name);
        if (it == factories_.end()) {
            return nullptr;
        }
        return it->second();
    }

    bool has(const std::string& name) const {
        return factories_.count(name) > 0;
    }

    void list_all() const {
        std::cout << "Available kernels:" << std::endl;
        for (const auto& kv : factories_) {
            std::cout << "  --" << kv.first << std::endl;
        }
    }

    const std::map<std::string, factory_fn>& all() const {
        return factories_;
    }

private:
    kernel_registry() = default;
    std::map<std::string, factory_fn> factories_;
};

// ============================================================================
// Registration macro
// ============================================================================

#define REGISTER_KERNEL(kernel_class)                                            \
    static bool kernel_class##_registered = []() {                               \
        bench_kernel::kernel_registry::instance().register_kernel(               \
            kernel_class().name(),                                               \
            []() -> bench_kernel::kernel_ptr {                                   \
                return std::make_shared<kernel_class>();                          \
            });                                                                  \
        return true;                                                             \
    }();

}  // namespace bench_kernel
