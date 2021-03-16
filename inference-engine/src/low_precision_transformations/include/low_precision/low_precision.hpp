// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/layer_transformation.hpp"


namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API LowPrecision;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::LowPrecision: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LowPrecision(const LayerTransformation::Params = LayerTransformation::Params());
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    LayerTransformation::Params params;
};
