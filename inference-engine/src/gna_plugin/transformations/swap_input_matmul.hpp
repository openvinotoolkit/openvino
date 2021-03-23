// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API SwapInputMatMul;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::SwapInputMatMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMul();
};
