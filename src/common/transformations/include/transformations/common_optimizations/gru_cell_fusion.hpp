// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GRUCellFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief GRUCellFusion transformation replaces a sequence of
 * operations with GRUCell op.
 *
 * If BiasAdds are not present in the pattern, then
 * Constants with zero values will be created to match the specification.
 *
 * Supported activations: Relu, Sigmoid, Tanh
 * Clip attribute is not supported.
 * Linear_before_reset attribute is not supported.
 * Supported weights formats: zr, rz
 *
 */

class ov::pass::GRUCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GRUCellFusion", "0");
    GRUCellFusion();
};
