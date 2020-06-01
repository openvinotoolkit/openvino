// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API CommonOptimizations;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::CommonOptimizations: public ngraph::pass::FunctionPass {
public:
    explicit CommonOptimizations() : FunctionPass() {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
