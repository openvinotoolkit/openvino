// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"
#include "openvino/core/except.hpp"

#include <memory>

namespace cldnn {

// This class is supposed to be compatible with std::optional, so can be removed after migration to c++17
template<typename T>
class optional_value {
public:
    optional_value() {}
    ~optional_value() {}

    optional_value(const T& v) : storage(new T(v)) {}
    optional_value(T&& v) : storage(new T(std::move(v))) {}

    optional_value(optional_value<T>&& other) : storage(std::move(other.storage)) {}
    optional_value(const optional_value<T>& other) {
        storage = nullptr;
        if (other.has_value())
            storage.reset(new T(other.value()));
    }

    optional_value<T>& operator=(const optional_value<T>& other) {
        storage = nullptr;
        if (other.has_value())
            storage.reset(new T(other.value()));
        return *this;
    }

    optional_value<T>& operator=(optional_value<T>&& other) {
        storage = std::move(other.storage);
        return *this;
    }

    optional_value<T>& operator=(const T& value) {
        storage.reset(new T(value));
        return *this;
    }

    optional_value<T>& operator=(T&& value) {
        storage.reset(new T(std::move(value)));
        return *this;
    }

    const T& value() const {
        OPENVINO_ASSERT(has_value(), "[GPU] Tried to get value from empty optional_value");
        return *storage;
    }

    const T& operator*() const {
        return value();
    }

    const T& value_or(const T& default_value) const noexcept {
        if (has_value())
            return *storage;
        return default_value;
    }

    bool has_value() const noexcept { return storage != nullptr; }
    operator bool() const noexcept { return storage != nullptr; }

private:
    std::unique_ptr<T> storage = nullptr;
};

using optional_data_type = optional_value<data_types>;
using optional_layout = optional_value<layout>;

}  // namespace cldnn
