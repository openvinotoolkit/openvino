// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveUselessConvertLike;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief RemoveUselessConvertLike transformation removes useless ConvertLike operations when input type already
 * matches out element type.
 *
 * PyTorch FE generates models with extra ConvertLike nodes and this makes more difficult pattern matching, especially
 * for precision sensitive nodes.
 */

class ov::pass::RemoveUselessConvertLike : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveUselessConvertLike", "0");
    RemoveUselessConvertLike();
};
