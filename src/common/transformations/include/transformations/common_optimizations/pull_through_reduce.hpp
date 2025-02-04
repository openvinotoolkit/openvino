// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PullThroughReduce;
class TRANSFORMATIONS_API PullUnsqueezeThroughReduce;
class TRANSFORMATIONS_API PullReshapeThroughReduce;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief PullUnsqueezeThroughReduce transformation
 * The transformation pulls Unsqueeze operator though Reduce ops if possible.
 * In the further processing such Unsqueeze can be often skipped as nop.
 */
class ov::pass::PullUnsqueezeThroughReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PullUnsqueezeThroughReduce");
    PullUnsqueezeThroughReduce();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief PullReshapeThroughReduce transformation
 * The transformation pulls Reshape operator though Reduce ops if possible.
 * In the further processing such Reshape can be often skipped as nop.
 */
class ov::pass::PullReshapeThroughReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PullReshapeThroughReduce");
    PullReshapeThroughReduce();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief PullThroughReduce transformation
 * The transformation pulls Reshape or Unsqueeze operators though Reduce ops if possible.
 * In the further processing such Reshape/Unsqueeze can be often skipped as nop.
 */
class ov::pass::PullThroughReduce : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("PullThroughReduce");
    PullThroughReduce() {
        add_matcher<ov::pass::PullUnsqueezeThroughReduce>();
        add_matcher<ov::pass::PullReshapeThroughReduce>();
    }
};
