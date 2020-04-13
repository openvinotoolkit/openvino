// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertOpSet1ToLegacy);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOpSet1ToLegacy: public ngraph::pass::FunctionPass, public ngraph::pass::PassParam {
public:
    explicit ConvertOpSet1ToLegacy(const PassParam::param_callback & callback = PassParam::getDefaultCallback())
             : FunctionPass(), PassParam(callback) {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
