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
 * @brief Add `meanImage` preprocessing to input nodes
 */
class AddMeanImage : public ov::pass::MatcherPass {
public:
    using MeanMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

    OPENVINO_RTTI("AddMeanImage", "0");
    explicit AddMeanImage(const MeanMap& inputInfoMap);
};

}  // namespace pass
}  // namespace ov
