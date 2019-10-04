// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include <details/ie_exception.hpp>

namespace vpu {

class AutoScope final {
public:
    explicit AutoScope(const std::function<void()>& func) : _func(func) {}

    ~AutoScope() {
        if (_func != nullptr) {
            _func();
        }
    }

    void callAndRelease() {
        if (_func != nullptr) {
            _func();
            _func = nullptr;
        }
    }

    void release() {
        _func = nullptr;
    }

    AutoScope(const AutoScope& other) = delete;
    AutoScope& operator=(const AutoScope&) = delete;

private:
    std::function<void()> _func;
};

}  // namespace vpu
