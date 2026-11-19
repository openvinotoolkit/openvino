// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Feeds ov::op::internal::GatedResidual (READ / FINAL_MIX) its weights compressed: the decompression subgraphs
 * of W_down, W_up and W_inject are replaced by the stored constants, scales and zero points.
 *
 * Each weight is a decompression subgraph of a stored constant Wq (a type from `supported_weights_types`), an optional
 * zero point zp and a scale s:
 *
 *   Before:                                        After:
 *
 *   Wq -> Convert -> [Subtract(zp)] -> Multiply(s) [-> Reshape]     Wq [N, K], s [N, G], [zp [N, G]]
 *                          |                                                 |
 *   residual, gain, W_down, W_up [, W_inject]            residual, gain, Wq_down, Wq_up, s_down, s_up
 *                          |                                [, zp_down, zp_up] [, Wq_inject, s_inject [, zp_inject]]
 *                    GatedResidual                                     GatedResidual (compressed form)
 *
 * The stored constants are taken as they are. The op is converted only if every weight is a compressed one of the same
 * type, its zero point (if any) has the shape of its scale, and all of them have zero points or none do (the zero
 * points of the op are all-or-nothing). The op is kept as it is when its weights are not compressed or are transposed
 * (a Transpose of the weights is fused into the MatMul before, so it is not expected), or when the backend rejects the
 * compressed op.
 */
class TRANSFORMATIONS_API ConvertGatedResidualToCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGatedResidualToCompressed");

    /// Whether the backend runs the converted op.
    using SupportsPredicate = std::function<bool(const std::shared_ptr<ov::op::internal::GatedResidual>&)>;

    /// @param supported_weights_types Types of the stored weights (and their zero points) the backend accepts.
    /// @param supports_config Asked about the converted op; returning false keeps the original one.
    ConvertGatedResidualToCompressed(const ov::element::TypeVector& supported_weights_types,
                                     SupportsPredicate supports_config = nullptr);
};

}  // namespace ov::pass
