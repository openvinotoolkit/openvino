// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief ShapeOfConstFolding constantfolds ShapeOf->Constant subgraph
 */
class TRANSFORMATIONS_API ShapeOfConstFolding : public MatcherPass {
public:
    OPENVINO_RTTI("ShapeOfConstFolding", "0");
    ShapeOfConstFolding();
};

}  // namespace pass
}  // namespace ov
