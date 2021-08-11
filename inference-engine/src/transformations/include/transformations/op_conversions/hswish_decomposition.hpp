// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API HSwishDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishDecomposition transformation into sub-graph x * (min(Relu(x + 3), 6) * const(1/6).
 */
class ov::pass::HSwishDecomposition: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    HSwishDecomposition();
};
