// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API NormalizeL2Fusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief NormalizeL2Fusion transformation replaces sub-graphs:
 * x/(sqrt(max(reduce_sum(x[j0, ..., jN]**2, axes), eps))
 * x/(sqrt(add(reduce_sum(x[j0, ..., jN]**2, axes), eps))
 * x/(pow(max(reduce_sum(x[j0, ..., jN]**2, axes), eps), 0.5)
 * x/(pow(add(reduce_sum(x[j0, ..., jN]**2, axes), eps), 0.5)
 * x*(pow(max(reduce_sum(x[j0, ..., jN]**2, axes), eps), -0.5)
 * x*(pow(add(reduce_sum(x[j0, ..., jN]**2, axes), eps), -0.5)
 * with a NormalizeL2(x, axes, eps, eps_mode[MAX|ADD]) op
 */
class ngraph::pass::NormalizeL2Fusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("NormalizeL2Fusion", "0");
    NormalizeL2Fusion();
};
