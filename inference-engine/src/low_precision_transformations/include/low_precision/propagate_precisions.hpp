// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <low_precision/lpt_visibility.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PropagatePrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::PropagatePrecisions : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
