// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeNonZero(std::shared_ptr<ngraph::Node> nonZero);

}  // namespace vpu
