// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/reshape.hpp"

namespace ov {
namespace builder {
namespace subgraph {

Reshape::Reshape() : isEmpty(true), special_zero(true) {
}

Reshape::Reshape(const std::vector<size_t>& values, const bool special_zero) :
    isEmpty(values.empty()),
    values(values),
    special_zero(special_zero) {
}

bool Reshape::empty() const noexcept {
    return isEmpty;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
