// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <vpu/model/base.hpp>

namespace vpu {

class BlobSerializer final {
public:
    template <typename T>
    int append(const T& val) {
        auto curPos = _data.size();

        _data.insert(
            _data.end(),
            reinterpret_cast<const char*>(&val),
            reinterpret_cast<const char*>(&val) + sizeof(val));

        return checked_cast<int>(curPos);
    }

    template <typename T>
    void overWrite(int pos, const T& val) {
        auto uPos = checked_cast<size_t>(pos);
        std::copy_n(reinterpret_cast<const char*>(&val), sizeof(val), _data.data() + uPos);
    }

    // Overwrites `uint32_t` value in `_data` at the position `pos`
    // to the size of the tail from `pos` to the end of `_data`.
    void overWriteTailSize(int pos) {
        auto uPos = checked_cast<size_t>(pos);
        IE_ASSERT(uPos < _data.size());
        auto size = checked_cast<uint32_t>(_data.size() - uPos);
        std::copy_n(reinterpret_cast<const char*>(&size), sizeof(uint32_t), _data.data() + uPos);
    }

    int size() const { return checked_cast<int>(_data.size()); }

    const char* data() const { return _data.data(); }

private:
    std::vector<char> _data;
};

}  // namespace vpu
