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
 * @brief Feeds GatedResidual compressed weights, replacing the decompression subgraphs by stored constants, scales
 * and zero points.
 *
 *   Wq [N, K | N, G, K/G] -> Convert -> [Subtract(zp)] -> Multiply(s) [-> Reshape]
 *     =>  Wq [N, K], s [N, G] [, zp [N, G]]   (inputs: ov::op::internal::GatedResidual::ReadLayout)
 *
 * Converted only if all weights are compressed, not transposed, of one type, with zero points on all or none, and the
 * backend accepts the op. The scales must be floating point, [N, G] or one element; the zero points of the weight type,
 * shaped like the scale or one element. Single-element scales and zero points are kept as they are.
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
