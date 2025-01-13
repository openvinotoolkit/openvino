// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSShapeOfForward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSShapeOfForward transformation sinks Transpose through ShapeOf in the forward direction.
 *
 * It replaces:
 *
 *   +---------+
 *   |Transpose|
 *   +---------+
 *        |
 *        v
 *   +---------+
 *   | ShapeOf |
 *   +---------+
 *
 * with the following:
 *
 *   +---------+
 *   | ShapeOf |
 *   +---------+
 *       |
 *       v
 *    +------+
 *    |Gather|
 *    +------+
 *
 */
class ov::pass::transpose_sinking::TSShapeOfForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("TSShapeOfForward", "0", ov::pass::transpose_sinking::TSForwardBase);
    TSShapeOfForward();
};
