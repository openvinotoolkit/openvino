// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace layout {
namespace utils {

std::vector<int64_t> find_permutation(const Layout& src_layout,
                                      const PartialShape& src_shape,
                                      const Layout& dst_layout);
Layout apply_permutation(const Layout& src_layout, const std::vector<uint64_t>& dims);

}  // namespace utils
}  // namespace layout
}  // namespace ov
