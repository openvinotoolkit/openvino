// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API OptimizerGatherND;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Optimize GatherND by replacing it with Reshape and Gather
 */
class ov::pass::OptimizerGatherND : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("OptimizerGatherND", "0");
    OptimizerGatherND();
};
