// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace SubgraphsDumper {
const std::shared_ptr<ngraph::Node> clone_with_new_inputs(const std::shared_ptr<ngraph::Node> &node);
}  // namespace SubgraphsDumper
