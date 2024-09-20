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

class TRANSFORMATIONS_API AUGRUCellFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief AUGRUCellFusion transformation replaces a sequence of
 * operations with AUGRUCell op.
 *
 * Supported activations: 1st is Sigmoid, 2nd is Tanh
 * Clip attribute is not supported.
 * Linear_before_reset attribute is not supported.
 * Supported weights format: 'rzh'
 *
 */

class ov::pass::AUGRUCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AUGRUCellFusion", "0");
    AUGRUCellFusion();
};
