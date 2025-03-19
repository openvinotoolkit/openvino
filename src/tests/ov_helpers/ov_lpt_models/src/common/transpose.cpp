// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/transpose.hpp"

namespace ov {
namespace builder {
namespace subgraph {

Transpose::Transpose() : isEmpty(true) {
}

Transpose::Transpose(const std::vector<size_t>& values) :
    isEmpty(values.empty()),
    values(values) {
}

bool Transpose::empty() const noexcept {
    return isEmpty;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
