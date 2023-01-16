// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

namespace ngraph {
namespace builder {
namespace subgraph {

class Reshape {
public:
    Reshape();
    Reshape(const std::vector<size_t>& values, const bool special_zero = true);
    bool empty() const noexcept;

    std::vector<size_t> values;
    bool special_zero;
private:
    bool isEmpty;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
