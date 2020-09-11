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

class TRANSFORMATIONS_API ReshapeMatMul;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeMatMul: public ngraph::pass::MatcherPass {
public:
    ReshapeMatMul();
};