// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

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
    OPENVINO_RTTI("LogSoftmaxDecomposition", "0");
    LogSoftmaxDecomposition();
};
