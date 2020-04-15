// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

#include <vector>
#include <memory>

namespace ngraph {
namespace pass {

class DynamicToStaticShape : public FunctionPass {
public:
    DynamicToStaticShape() = default;

    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

private:
    bool validateStaticShapes(std::shared_ptr<ngraph::Function> function) const;
};

}  // namespace pass
}  // namespace ngraph
