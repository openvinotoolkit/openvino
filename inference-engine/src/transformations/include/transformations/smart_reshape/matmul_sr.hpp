// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReshapeAMatMul;
class TRANSFORMATIONS_API ReshapeBMatMul;
class TRANSFORMATIONS_API TransposeMatMul;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeAMatMul : public ngraph::pass::MatcherPass {
public:
    ReshapeAMatMul();
};
class ngraph::pass::ReshapeBMatMul : public ngraph::pass::MatcherPass {
public:
    ReshapeBMatMul();
};
class ngraph::pass::TransposeMatMul : public ngraph::pass::MatcherPass {
public:
    TransposeMatMul();
};