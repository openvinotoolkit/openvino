// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "ngraph/pass/graph_rewrite.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief Add `meanValue` or `meanImage` preprocessing to input nodes
 */
class AddMeanSubtract : public ngraph::pass::MatcherPass {
public:
    using MeanMap = std::map<std::string, std::shared_ptr<ngraph::op::v0::Constant>>;

    OPENVINO_RTTI("AddMeanSubtract");
    explicit AddMeanSubtract(const MeanMap& inputInfoMap);
};

}  // namespace pass
}  // namespace ov
