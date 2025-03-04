// Copyright (C) 2022-2023 Intel Corporation
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

class TRANSFORMATIONS_API TSSplitBackward;
class TRANSFORMATIONS_API TSSplitForward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSSplitForward transformation sinks Transpose through Split, VariadicSplit operations
 * in the forward direction.
 */
class ov::pass::transpose_sinking::TSSplitForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSSplitForward", "0", ov::pass::transpose_sinking::TSForwardBase);
    TSSplitForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSSplitBackward transformation sinks Transpose through Split, VariadicSplit operations
 * in the backward direction.
 */
class ov::pass::transpose_sinking::TSSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::pass::TSSplitBackward");
    TSSplitBackward();
};
