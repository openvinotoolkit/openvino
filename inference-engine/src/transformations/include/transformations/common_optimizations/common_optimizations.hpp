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

class INFERENCE_ENGINE_API_CLASS(CommonOptimizations);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::CommonOptimizations: public ngraph::pass::FunctionPass {
public:
    explicit CommonOptimizations() : FunctionPass() {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
