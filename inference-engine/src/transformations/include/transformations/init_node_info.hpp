// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(InitNodeInfo);

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     InitNodeInfo transformation helps to set runtime info attributes in a single place.
 *     Every runtime info attribute that needs to be initialized should be registered
 *     in run_on_function method. Also do not forget to override init methods for registered
 *     attribute.
 *     This transformations should be called first in transformation pipeline. If attrbute was
 *     already set initialization will be skipped for this node.
 */

class ngraph::pass::InitNodeInfo: public ngraph::pass::FunctionPass {
public:
    InitNodeInfo() : FunctionPass() {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
