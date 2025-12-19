// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertWeightCompressedConv1x1ToMatmul;
class TRANSFORMATIONS_API ConvertWeightCompressedConv1x1ToMatmul_ActNotTran;

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

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertWeightCompressedConv1x1ToMatmul_ActNotTran transformation matches a weight-compressed
 *        Convolution with a 1x1 kernel and replaces it with a MatMul operation. The activation input of convolution is
 *        not transposed.
 *
 * The transformation identifies the following pattern:
 *
 *                       +---------+    +-----------+    +------+  last two dims are 1
 *                       | Weights |    | ZeroPoint |    |Scale |
 *                       | (5D)    |    | (5D)      |    | (5D) |  5D or 4D with
 *                       +---------+    +-----------+    +------+  reshape(w), unsqueeze(zp), unsqueeze(scale) to 5D
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
 *           |                        |                     |
 *           |                        v                     |
 *           |                   +----------+               |
 *           |                   | Multiply |<--------------+
 *           |                   +----------+
 *           |                        |
 *           |                        v
 *           |                  +-----------+
 *           |                  |  Reshape  | To shape contains 4 dims, last two are 1
 *           |                  +-----------+
 *           |                        |
 *           |                        v
 *           +----------------->|Convolution|
 *                              |  (1x1)    |
 *                              +-----------+
 *
 * and replaces it with:
 *
 *                       +---------+    +-----------+    +------+  Removed last two dims of 1
 *                       | Weights |    | ZeroPoint |    |Scale |
 *                       | (3D)    |    | (3D)      |    | (3D) |  3D or 2D with
 *                       +---------+    +-----------+    +------+  reshape(w), unsqueeze(zp), unsqueeze(scale) to 3D
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
 *     +-----------+                  v                     |
 *     | Transpose |             +----------+               |
 *     +-----------+             | Multiply |<--------------+
 *           |                   +----------+
 *           |                        |
 *           |                        v
 *           |                  +-----------+
 *           |                  |  Reshape  | To shape contains 2 dims, removed last two dims of 1
 *           |                  +-----------+
 *           |                        |
 *           |                        v
 *           |                  +-----------+
 *           +----------------->|  MatMul   |
 *                              +-----------+
 *                                    |
 *                                    v
 *                              +-----------+
 *                              | Transpose |
 *                              +-----------+
 */
class ov::pass::ConvertWeightCompressedConv1x1ToMatmul_ActNotTran : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertWeightCompressedConv1x1ToMatmul_ActNotTran");
    explicit ConvertWeightCompressedConv1x1ToMatmul_ActNotTran(const element::TypeVector& supported_precisions,
                                                               const element::TypeVector& unsupported_precisions);
};
