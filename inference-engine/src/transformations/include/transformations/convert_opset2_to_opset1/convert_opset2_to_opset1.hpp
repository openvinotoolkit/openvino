// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertOpSet2ToOpSet1;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOpSet2ToOpSet1: public ngraph::pass::FunctionPass, public ngraph::pass::PassParam {
public:
    explicit ConvertOpSet2ToOpSet1(const PassParam::param_callback & callback = PassParam::getDefaultCallback())
             : FunctionPass(), PassParam(callback) {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
