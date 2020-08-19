// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API NopElimination;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::NopElimination: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<Function>) override;
};
