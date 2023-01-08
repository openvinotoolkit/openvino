// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GatherNegativeConstIndicesNormalize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief GatherNegativeConstIndicesNormalize checks if indices value is negative scalar and
 * normalizes it using ShapeOf->Add->Cast subgraph.
 * We need to remove this transformation after adding support of negative indices in
 * future version of Gather operation.
 */
class ov::pass::GatherNegativeConstIndicesNormalize : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherNegativeConstIndicesNormalize", "0");
    GatherNegativeConstIndicesNormalize();
};

namespace ngraph {
namespace pass {
using ov::pass::GatherNegativeConstIndicesNormalize;
}  // namespace pass
}  // namespace ngraph
