// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/markup_precisions.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API LowPrecision;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::LowPrecision : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LowPrecision(
        const std::vector<OperationPrecisionRestriction>& restrictions = {},
        // TODO: debug only
        const LayerTransformation::Params = LayerTransformation::Params());
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    static bool isFunctionQuantized(const std::shared_ptr<ngraph::Function>& function);
protected:
    std::vector<OperationPrecisionRestriction> restrictions;
    // remove
    LayerTransformation::Params params;
};
