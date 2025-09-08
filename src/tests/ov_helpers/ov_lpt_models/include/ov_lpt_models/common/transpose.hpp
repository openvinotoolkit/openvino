// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

namespace ov {
namespace builder {
namespace subgraph {

class Transpose {
public:
    Transpose();
    Transpose(const std::vector<size_t>& values);
    bool empty() const noexcept;

    std::vector<size_t> values;
private:
    bool isEmpty;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
