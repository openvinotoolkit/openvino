// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief Add `stdScale` preprocessing to input nodes
 */
class AddScale : public ov::pass::MatcherPass {
public:
    using ScaleMap = std::map<std::string, std::shared_ptr<ngraph::op::v0::Constant>>;

    OPENVINO_RTTI("AddStdScale", "0");
    explicit AddScale(const ScaleMap& inputInfoMap);
};

}  // namespace pass
}  // namespace ov
