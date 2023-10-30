// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief This transformation enables constfoldability for ShapeOf nodes that was
 * disabled by DisableShapeOfConstantFolding.
 */
class TRANSFORMATIONS_API EnableShapeOfConstantFolding : public MatcherPass {
public:
    OPENVINO_RTTI("EnableShapeOfConstantFolding", "0");
    EnableShapeOfConstantFolding(bool check_shape = true);
};

}  // namespace pass
}  // namespace ov
