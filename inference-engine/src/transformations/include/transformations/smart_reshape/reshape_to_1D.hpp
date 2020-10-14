// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ReshapeTo1D;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeTo1D : public ngraph::pass::MatcherPass {
public:
    ReshapeTo1D();
};
