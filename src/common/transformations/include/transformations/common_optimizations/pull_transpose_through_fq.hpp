// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <openvino/core/visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API PullTransposeThroughFQUp;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::PullTransposeThroughFQUp: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullTransposeThroughFQUp();
};
