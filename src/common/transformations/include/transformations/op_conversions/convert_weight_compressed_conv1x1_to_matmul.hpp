// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertWeightCompressedConv1x1ToMatmul;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertWeightCompressedConv1x1ToMatmul transformation matches a weight-compressed
 *        Convolution with a 1x1 kernel and replaces it with a MatMul operation.
 *
 * The transformation identifies the following pattern:
 *
 *                       +---------+    +-----------+    +------+
 *                       | Weights |    | ZeroPoint |    |Scale |
 *                       +---------+    +-----------+    +------+
 *                             |              |             |
 *                             v              v             |
 *                         +-------+      +-------+         |
 *                         |Convert|      |Convert|         |
 *                         +-------+      +-------+         |
 *                              |             |             |
 *                              +-----+  +----+             |
 *                                    |  |                  |
 *     +------------+                 v  v                  |
 *     | Activation |             +--------+                |
 *     +------------+             |Subtract| (optional)     |
 *           |                    +--------+                |
 *           v                        |                     |
 *     +-------------+                v                     |
 *     | Transpose/  |           +----------+               |
 *     | Reshape     |           | Multiply |<--------------+
 *     +-------------+           +----------+
 *           |                        |
 *           |                  *-----------+
 *           |                  | Reshape   | (optional)
 *           |                  +-----------+
 *           |                        |
 *           |                        v
 *           |                  +-----------+
 *           +----------------->|Convolution|
 *                              |  (1x1)    |
 *                              +-----------+
 *                                    |
 *                                    v
 *                               +----------+
 *                               |Add (Bias)| (optional)
 *                               +----------+
 *                                    |
 *                                    v
 *                              +-----------+
 *                              |  Convert  | (optional)
 *                              +-----------+
 *                                    |
 *                                    v
 *                              +------------+
 *                              | Transpose/ |
 *                              |  Reshape   |
 *                              +------------+
 *
 * and replaces it with:
 *
 *    +------------+
 *    | Activation |
 *    +------------+
 *           |
 *           |       +---------+   +-----------+    +------+
 *           |       | Weights |   | ZeroPoint |    |Scale |
 *           |       +---------+   +-----------+    +------+
 *           |            |              |             |
 *           |            v              v             |
 *           |        +-------+      +-------+         |
 *           |        |Convert|      |Convert|         |
 *           |        +-------+      +-------+         |
 *           |             |             |             |
 *           |             +-----+  +----+             |
 *           |                   |  |                  |
 *           |                   v  v                  |
 *           |               +--------+                |
 *           |               |Subtract| (optional)     |
 *           |               +--------+                |
 *           |                    |                    |
 *           |                    v                    |
 *           |              +----------+               |
 *           |              | Multiply |<--------------+
 *           |              +----------+
 *           |                   |
 *           |                   v
 *           |             *----------+
 *           |             | Reshape  | (optional)
 *           |             +----------+
 *           |                   |
 *           |                   v
 *           |               +--------+
 *           +-------------> | MatMul |
 *                           +--------+
 *                               |
 *                               v
 *                          +----------+
 *                          |Add (Bias)| (optional)
 *                          +----------+
 *                                |
 *                                v
 *                          +-----------+
 *                          |  Convert  | (optional)
 *                          +-----------+
 *                                |
 *                                v
 *                          +------------+
 *                          |  Reshape   | (optional)
 *                          +------------+
 */

class ov::pass::ConvertWeightCompressedConv1x1ToMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertWeightCompressedConv1x1ToMatmul");
    ConvertWeightCompressedConv1x1ToMatmul();
};
