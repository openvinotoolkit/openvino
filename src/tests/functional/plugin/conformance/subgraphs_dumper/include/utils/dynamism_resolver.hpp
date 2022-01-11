// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace SubgraphsDumper {
// Copy from serialization transformation pass
void resolve_dynamic_shapes(const std::shared_ptr<ngraph::Function>& f);

} // namespace SubgraphsDumper