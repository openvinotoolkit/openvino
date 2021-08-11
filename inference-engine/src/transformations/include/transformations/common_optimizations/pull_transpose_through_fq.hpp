// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PullTransposeThroughFQUp;

}  // namespace pass
}  // namespace ov

class ov::pass::PullTransposeThroughFQUp: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullTransposeThroughFQUp();
};
