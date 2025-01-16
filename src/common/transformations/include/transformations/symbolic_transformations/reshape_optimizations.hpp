// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ReshapeOptimizations;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Searches for Flatten-like Reshape operations and simplifies 2nd input of such Reshape using special zero
 * feature.
 * The transformation works in 2 cases:
 * 1. all in/out dims are static, or we can match them via the symbols.
   2. only one out dim doesn't have the corresponding input static dim,
      and we can't match it using symbols. Besides that the output shape must not contain zero dims,
      because then value -1 in 2nd input to Reshape op can't guarantee an unambiguous determination of the remaining dim
 value.

      for example:
            Before:
            +-------------+    +----------+
            |data         |    | Concat   |
            |shape: (0, 0)|    | shape (2)|<- the values might be determined in runtime
            +-----+-------+    +-----+----+   the empty data tensor can be reshaped to
                  |                  |        any other empty shape, e.g. (0, 800)
            +-----v---------------+  |
            | Reshape             |  |
            | shape (0,-1)        <--+
            | special zero = False|
            +---------------------+

            After:
                               +---------------+
           +--------------+    | Constant      |
           | data         |    | shape (2)     |
           | shape: (0, 0)|    | values (0, -1)|<- -1 means copy the corresponding input dim
           +-----+--------+    +-------+-------+
                 |                     |
            +----v----------------+    |
            | Reshape             <----+
            | shape (0,0)         |  <- it might cause inconsistency
            | special zero = True |
            +---------------------+
 */
class ov::pass::ReshapeOptimizations : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeOptimizations", "0");
    ReshapeOptimizations();
};
