// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// Matches v17::GroupedMatMul in both legal input arities:
//   - 2 inputs (3D x 3D form):  GroupedMatMul(data, compressed_weights)
//   - 3 inputs (2D x 3D form):  GroupedMatMul(data, compressed_weights, offsets)
// and rewrites it into ov::op::internal::GroupedMatMulCompressed.
class TRANSFORMATIONS_API ConvertGroupedMatMulToGroupedMatMulCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMulToGroupedMatMulCompressed");
    explicit ConvertGroupedMatMulToGroupedMatMulCompressed(
        const std::vector<ov::element::Type>& supported_weights_types);
};

}  // namespace ov::pass
