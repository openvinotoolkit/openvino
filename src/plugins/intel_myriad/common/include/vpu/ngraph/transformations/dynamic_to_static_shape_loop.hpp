// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"

#include <memory>

namespace vpu {

void validateLoop(const ngraph::Node& node);
void dynamicToStaticShapeLoop(std::shared_ptr<ngraph::Node> node);

}  // namespace vpu
