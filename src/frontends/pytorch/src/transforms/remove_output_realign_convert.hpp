// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * @interface RemoveOutputRealignConvert
 * @brief Removes a Convert/ConvertLike marked by mark_type_realign_convert (see
 * type_realign_convert.hpp) when it feeds only Result node(s), letting the higher precision
 * value computed upstream reach the model output directly instead of being narrowed down.
 * Always clears the marker so it never leaks into the exported IR.
 * @ingroup ov_frontends
 */
class RemoveOutputRealignConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::RemoveOutputRealignConvert");
    RemoveOutputRealignConvert();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
