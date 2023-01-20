// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReduceL2Decomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Decomposes ReduceL2 into sqrt(ReduceSum(x * x)).
 */
class ov::pass::ReduceL2Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceL2Decomposition", "0");
    ReduceL2Decomposition();
};

namespace ngraph {
namespace pass {
using ov::pass::ReduceL2Decomposition;
}  // namespace pass
}  // namespace ngraph
