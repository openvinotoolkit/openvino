// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

    class TRANSFORMATIONS_API LogSoftmaxDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief LogSoftmaxDecomposition transformation into sub-graph x - log(reduce_sum(exp(x), axis)).
 */
class ov::pass::LogSoftmaxDecomposition : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LogSoftmaxDecomposition();
};
