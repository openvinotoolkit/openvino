//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/span.hpp"

#include <optional>
#include <type_traits>
#include <vector>

namespace intel_npu::vm {

/// A vector-like container that can either own its data (by making a copy of it) or reference external data without
/// taking ownership. The template parameter T represents the type of the elements in the vector.
template <typename T>
class ManagedVector {
    static_assert(std::is_trivially_copyable_v<T>, "ManagedVector requires a trivially copyable element type");

    std::optional<std::vector<T>> _ownedData;  // Optional owned copy of the data, if this instance owns its data
    Span<T> _data;                             // The data span, either referencing the owned data or external data

public:
    ManagedVector() = default;

    /// Constructs a ManagedVector that either owns a copy of the provided data or references it directly, depending on
    /// the value of copyData. If copyData is true, the constructor makes an owned copy of the data; otherwise, it just
    /// references the provided data without taking ownership. The caller must ensure that the provided data remains
    /// valid for the lifetime of the ManagedVector if copyData is false.
    ManagedVector(Span<T> data, bool copyData = false) {
        if (copyData) {
            // Make an owned copy of the data if requested
            auto& ownedData = _ownedData.emplace();
            ownedData.reserve(data.size());
            ownedData.insert(ownedData.end(), data.begin(), data.end());
            _data = Span<T>(ownedData.data(), ownedData.size());
        } else {
            // Otherwise just reference the provided data directly
            _data = data;
        }
    }

    ManagedVector(const ManagedVector& other) {
        if (other._ownedData.has_value()) {
            // Deep-copy the owned data so each instance manages its own allocation
            _ownedData = other._ownedData;
            auto& ownedData = _ownedData.value();
            _data = Span<T>(ownedData.data(), ownedData.size());
        } else {
            // Share the same external data without taking ownership
            _data = other._data;
        }
    }

    ManagedVector(ManagedVector&& other) noexcept: _ownedData(std::move(other._ownedData)) {
        if (_ownedData.has_value()) {
            auto& ownedData = _ownedData.value();
            _data = Span<T>(ownedData.data(), ownedData.size());
        } else {
            _data = other._data;
        }
        other._data = Span<T>();
    }

    ManagedVector& operator=(const ManagedVector& other) {
        if (this != &other) {
            // Copy into temporaries first, to leave *this in a valid state if the vector copy fails
            std::optional<std::vector<T>> newOwnedData;
            Span<T> newData;

            if (other._ownedData.has_value()) {
                newOwnedData = other._ownedData;
                auto& ownedData = newOwnedData.value();
                newData = Span<T>(ownedData.data(), ownedData.size());
            } else {
                newData = other._data;
            }

            // Move the data from the temporaries into *this
            _ownedData = std::move(newOwnedData);
            _data = newData;
        }
        return *this;
    }

    ManagedVector& operator=(ManagedVector&& other) noexcept {
        if (this != &other) {
            _ownedData = std::move(other._ownedData);
            if (_ownedData.has_value()) {
                auto& ownedData = _ownedData.value();
                _data = Span<T>(ownedData.data(), ownedData.size());
            } else {
                _data = other._data;
            }
            other._data = Span<T>();
        }
        return *this;
    }

    ~ManagedVector() = default;

    /// Returns true if this ManagedVector owns its data, false if it is just referencing external data
    bool isOwned() const {
        return _ownedData.has_value();
    }

    /// Returns a Span representing the elements in the vector
    Span<const T> get() const {
        return Span<const T>(_data.begin(), _data.size());
    }

    /// Returns a Span representing the elements in the vector
    Span<T> get() {
        return _data;
    }
};

}  // namespace intel_npu::vm
