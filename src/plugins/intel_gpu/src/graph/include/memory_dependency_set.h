// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cldnn {

// Build dense restriction sets without shifting a sorted vector for every insertion.
// Small or sparse sets remain vectors; consumers still receive sorted, unique IDs.
class memory_dependency_set {
public:
    memory_dependency_set() = default;
    explicit memory_dependency_set(std::vector<uint32_t> values) : _values(std::move(values)) {}

    bool contains(uint32_t id) const {
        if (_bits.empty())
            return std::binary_search(_values.begin(), _values.end(), id);
        const size_t word = id / 64;
        return word >= _first_word && word - _first_word < _bits.size() && (_bits[word - _first_word] & (uint64_t{1} << (id % 64))) != 0;
    }

    void insert(uint32_t id) {
        if (!_bits.empty()) {
            const size_t word = id / 64;
            const size_t first = std::min(_first_word, word);
            const size_t end = std::max(_first_word + _bits.size(), word + 1);
            // An outlying ID must not turn a sparse set into a large bitmap.
            if ((end - first) * sizeof(uint64_t) > (_count + 1) * sizeof(uint32_t)) {
                values();
            } else {
                if (first < _first_word) {
                    _bits.insert(_bits.begin(), _first_word - first, 0);
                    _first_word = first;
                }
                _bits.resize(end - first, 0);
                auto& bits = _bits[word - _first_word];
                const auto mask = uint64_t{1} << (id % 64);
                _count += (bits & mask) == 0;
                bits |= mask;
                return;
            }
        }

        auto it = std::lower_bound(_values.begin(), _values.end(), id);
        if (it != _values.end() && *it == id)
            return;
        _values.insert(it, id);

        // Reconsider sparse sets only at geometrically increasing sizes.
        const auto size = _values.size();
        if (size < 64 || (size & (size - 1)) != 0)
            return;
        const size_t first = _values.front() / 64;
        const size_t words = _values.back() / 64 - first + 1;
        if (words * sizeof(uint64_t) > size * sizeof(uint32_t))
            return;

        std::vector<uint64_t> bits(words, 0);
        for (auto value : _values)
            bits[value / 64 - first] |= uint64_t{1} << (value % 64);
        _bits.swap(bits);
        _first_word = first;
        _count = size;
        std::vector<uint32_t>().swap(_values);
    }

    // Materialize once at the compiler/runtime boundary, and release construction storage.
    const std::vector<uint32_t>& values() {
        if (!_bits.empty()) {
            _values.reserve(_count);
            for (size_t i = 0; i < _bits.size(); ++i) {
                auto bits = _bits[i];
                for (uint32_t bit = 0; bits != 0; ++bit, bits >>= 1) {
                    if ((bits & 1) != 0)
                        _values.push_back(static_cast<uint32_t>((_first_word + i) * 64 + bit));
                }
            }
            std::vector<uint64_t>().swap(_bits);
            _count = 0;
        }
        return _values;
    }

private:
    std::vector<uint32_t> _values;
    std::vector<uint64_t> _bits;
    size_t _first_word = 0;
    size_t _count = 0;
};

}  // namespace cldnn
