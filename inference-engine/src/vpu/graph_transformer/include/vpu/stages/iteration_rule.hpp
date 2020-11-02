// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/model/data_desc.hpp"

#include <map>
#include <tuple>
#include <utility>
#include <cstdint>

namespace vpu {

class IterationRule {
public:
    IterationRule(Dim new_axis, int32_t new_start, int32_t new_stride, int32_t new_end)
        : axis(new_axis), start(new_start), stride(new_stride), end(new_end) {}

    Dim axis = Dim::Invalid;
    int32_t start  =  0;
    int32_t stride =  1;
    int32_t end    = -1;

    bool operator<(const IterationRule& rhs) const {
        return key() < rhs.key();
    }

private:
    using Key = std::tuple<Dim, int32_t, int32_t, int32_t>;
    Key key() const {
        return std::make_tuple(axis, start, stride, end);
    }
};

using IterationComponents = std::map<std::pair<std::size_t, IterationRule>, std::size_t>;

}  // namespace vpu
