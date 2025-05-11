// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/layout.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace layout {
namespace utils {

// Example is NCHW to HWC. Need to calculate user's shape from (?, 3, 480, 640) to (480, 640, 3)
// src_layout shall be 'bigger' than 'dst_layout'
// Returns shape and layout after 'squeeze' (CHW). Next step will be to apply "find_permutation" CHW->HWC
std::tuple<PartialShape, Layout> find_squeeze(const Layout& src_layout,
                                              const PartialShape& src_shape,
                                              const Layout& dst_layout);

// Example is HWC to NCDHW. Needs also to calculate user's shape from (480, 640, 3) to (1, 3, 1, 480, 640)
// src_layout shall be 'smaller' than 'dst_layout'
// Returns shape, layout and number of axis for unsqueeze after 'unsqueeze'.
// In this example, function will return: Shape {1,1,480,640,3}, Layout "NDCHW", axis=2
// Next step will be to apply "find_permutation" NDCHW->NCDHW
std::tuple<PartialShape, Layout, size_t> find_unsqueeze(const Layout& src_layout,
                                                        const PartialShape& src_shape,
                                                        const Layout& dst_layout);

std::vector<int64_t> find_permutation(const Layout& src_layout,
                                      const PartialShape& src_shape,
                                      const Layout& dst_layout);
Layout apply_permutation(const Layout& src_layout, const std::vector<uint64_t>& dims);

bool is_compatible(const Layout& layout, const PartialShape& shape);

}  // namespace utils
}  // namespace layout
}  // namespace ov
