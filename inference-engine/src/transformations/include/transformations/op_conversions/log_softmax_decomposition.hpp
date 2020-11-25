// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API LogSoftmaxDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief LogSoftmaxDecomposition transformation into sub-graph x - log(reduce_sum(exp(x), axis)).
 */
class ngraph::pass::LogSoftmaxDecomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LogSoftmaxDecomposition();
};
