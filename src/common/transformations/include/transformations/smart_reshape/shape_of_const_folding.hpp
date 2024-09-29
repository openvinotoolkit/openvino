// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief ShapeOfConstFolding constantfolds ShapeOf->Constant subgraph
 */
class TRANSFORMATIONS_API ShapeOfConstFolding : public MatcherPass {
public:
    OPENVINO_RTTI("ShapeOfConstFolding", "0");
    ShapeOfConstFolding();
};

}  // namespace pass
}  // namespace ov
