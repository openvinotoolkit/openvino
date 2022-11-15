// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PullTransposeThroughFQUp;

}  // namespace pass
}  // namespace ov

class ov::pass::PullTransposeThroughFQUp : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PullTransposeThroughFQUp", "0");
    PullTransposeThroughFQUp();
};

namespace ngraph {
namespace pass {
using ov::pass::PullTransposeThroughFQUp;
}  // namespace pass
}  // namespace ngraph
