// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"

std::vector<std::int64_t> evaluateTargetShape(const ngraph::Output<ngraph::Node>& value);

namespace vpu {

void printTo(std::ostream& stream, const ngraph::NodeTypeInfo& object);

}  // namespace vpu
