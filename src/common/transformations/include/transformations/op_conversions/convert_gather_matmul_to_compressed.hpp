// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// ============================================================================
// ConvertGatherMatmulToGatherMatmulCompressed
// ============================================================================
//
// Folds the compressed-weights dequantization subgraph on GatherMatmul's
// weight input into a single GatherMatmulCompressed op with explicit
// scale (and optional zero-point) inputs.
//
// BEFORE:
//
//   activation   weights(u4/u8/i4/i8)   indices   bias
//        |              |                   |        |
//        |           Convert(f16/f32)       |        |
//        |              |                   |        |
//        |          (Subtract(zp))?         |        |
//        |              |                   |        |
//        |         Multiply(scale)          |        |
//        |              |                   |        |
//        |          (Reshape)?              |        |
//        |              |                   |        |
//        |         (Transpose)?             |        |
//        |              |                   |        |
//        |          (Convert)?              |        |
//        |              |                   |        |
//        +------GatherMatmul---------------+--------+
//                    |
//                  output
//
//
// AFTER:
//
//   activation   weights(u4/u8)   indices   bias   scale   (zp)?
//        |            |              |        |       |       |
//        +----GatherMatmulCompressed---------+-------+-------+
//                          |
//                        output
//
// ============================================================================
class TRANSFORMATIONS_API ConvertGatherMatmulToGatherMatmulCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGatherMatmulToGatherMatmulCompressed");

    using SupportsPredicate =
        std::function<bool(const std::shared_ptr<ov::op::internal::GatherMatmulCompressed>&, size_t, size_t, size_t)>;

    ConvertGatherMatmulToGatherMatmulCompressed(const std::vector<ov::element::Type>& supported_activation_types,
                                                const std::vector<ov::element::Type>& supported_weights_types,
                                                const SupportsPredicate& supports_config = nullptr,
                                                bool convert_u4zp_to_u8 = false);
};

}  // namespace ov::pass
