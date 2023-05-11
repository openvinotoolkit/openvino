// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface SetSoftmaxPorts
 * @brief The pass updates port descriptors in accordance with the Softmax reduction axis
 * @ingroup snippets
 */
class SetSoftmaxPorts: public ngraph::pass::MatcherPass {
public:
    SetSoftmaxPorts();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
