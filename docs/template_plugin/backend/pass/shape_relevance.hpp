// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/pass.hpp"

namespace ngraph {
namespace pass {
class ShapeRelevance : public FunctionPass {
public:
    ShapeRelevance() : FunctionPass() {}
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
}  // namespace pass
}  // namespace ngraph
