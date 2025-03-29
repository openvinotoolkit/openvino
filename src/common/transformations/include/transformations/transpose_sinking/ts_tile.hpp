// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSTileForward;
class TRANSFORMATIONS_API TSTileBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSTileForward transformation sinks Transpose through Tile in the forward direction.
 */
class ov::pass::transpose_sinking::TSTileForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSBinaryForward", "0", ov::pass::transpose_sinking::TSForwardBase);
    TSTileForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSTileBackward transformation sinks Transpose through Tile in the backward direction.
 */
class ov::pass::transpose_sinking::TSTileBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::pass::TSTileBackward");
    TSTileBackward();
};
