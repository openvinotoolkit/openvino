// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <ngraph/op/constant.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>

#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

class AddMeanSubtract;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Add `meanValue` or `meanImage` preprocessing to input nodes
 */
class ngraph::pass::AddMeanSubtract : public ngraph::pass::MatcherPass {
public:
    using MeanMap = std::map<std::string, std::shared_ptr<ngraph::op::v0::Constant>>;

    NGRAPH_RTTI_DECLARATION;
    explicit AddMeanSubtract(const MeanMap& inputInfoMap);
};
