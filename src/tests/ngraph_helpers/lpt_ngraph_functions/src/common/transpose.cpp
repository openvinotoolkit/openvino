// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/common/transpose.hpp"

namespace ngraph {
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
}  // namespace ngraph
