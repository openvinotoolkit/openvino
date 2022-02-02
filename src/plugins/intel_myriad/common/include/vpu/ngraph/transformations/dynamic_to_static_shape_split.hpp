// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"

namespace vpu {

void validateSplit(const ngraph::Node& node);
void dynamicToStaticShapeSplit(std::shared_ptr<ngraph::Node> target);

}  // namespace vpu
