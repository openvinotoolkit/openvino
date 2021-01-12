// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API MarkupAvgPoolPrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// Transformation is used to add customization options runtime
class ngraph::pass::low_precision::MarkupAvgPoolPrecisions : public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
