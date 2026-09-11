// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// Matches v17::GroupedMatMul in both legal input arities:
//   - 2 inputs (3D x 3D form):  GroupedMatMul(data, compressed_weights)
//   - 3 inputs (2D x 3D form):  GroupedMatMul(data, compressed_weights, offsets)
// and rewrites it into ov::op::internal::GroupedMatMulCompressed.
class TRANSFORMATIONS_API ConvertGroupedMatMulToGroupedMatMulCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMulToGroupedMatMulCompressed");

    // Called with the freshly built GroupedMatMulCompressed and its IC (K), OC (N) and the number of
    // quantization groups along IC. Returning false leaves the original op untouched.
    using SupportsPredicate =
        std::function<bool(const std::shared_ptr<ov::op::internal::GroupedMatMulCompressed>&, size_t, size_t, size_t)>;

    explicit ConvertGroupedMatMulToGroupedMatMulCompressed(
        const std::vector<ov::element::Type>& supported_weights_types,
        const SupportsPredicate& supports_config = nullptr);
};

}  // namespace ov::pass
