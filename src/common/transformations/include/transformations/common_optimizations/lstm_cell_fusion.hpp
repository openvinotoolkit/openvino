// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMCellFusion;
class TRANSFORMATIONS_API LSTMCellTfKerasFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellFusion transformation replaces a sequence of
 * operations with LSTMCell op.
 */
class ov::pass::LSTMCellFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellFusion", "0");
    LSTMCellFusion();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LSTMCellTfKerasFusion transformation replaces a sequence of
 * operations with LSTMCell op.
 * Substitute next subgraph
 *     +---+   +---+    +---+  +---+
 *     | X |   | W |    | H |  | R |
 *     +--++   ++--+    +--++  ++--+
 *        |     |          |    |
 *       +v-----v-+      +-v----v-+
 *       | MatMul |      | MatMul |
 *       +--------++    ++--------+
 *                 |    |
 *                 |    |
 *                +v----v+        +----+
 *                | Add  |        | B  |
 *                +------++   +---+----+
 *                        |   |
 *                      +-v---v-+
 *                      |  Add  |
 *                      +---+---+
 *                          |
 *                      +---v---+
 *                      | Split |
 *    +-----------+-----+-------+-+--------------+
 *    |           |               |              |
 *    |           |               |              |
 * +--v----+   +--v----+     +----v---+     +----v----+
 * |Sigmoid|   |Sigmoid|     |  Tanh  |     | Sigmoid |
 * +-------+   +-----+-+     +--+-----+     +-----+---+
 *         |         |          |                 |
 *         |         |          |                 |
 *         |         |          |                 |
 *       +-v-------+ |          |     +---+       |
 *       |Multiply <-+----------+   +-+ C |       |
 *       +--------++ |              | +---+       |
 *                |  |              |             |
 *                |  |       +------v-+           |
 *                |  +------->Multiply|           |
 *                |          +----+---+           |
 *                |               |               |
 *                |               |               |
 *               +v----+          |               |
 *               | Add <----------+               |
 *          +----+--+--+                          |
 *          |       |                             |
 *          |       |                             |
 *       +--v+      |   +------+            +-----v----+
 *       |C0 |      +---> Tanh +------------>Multiply  |
 *       +---+          +------+            +--+-------+
 *                                             |
 *                                             |
 *                                             |
 *                                           +-v--+
 *                                           | H0 |
 *                                           +----+
 *
 *   With subgraph
 * +---+      +---+      +---+     +---+        +---+       +---+
 * | X |      | H |      | C |     | W |        | R |       | B |
 * +-+-+      +-+-+      +-+-+     +-+-+        +-+-+       +-+-+
 *   |          |          |         |            |           |
 *   |          |          |  +------v-----+  +---v------+    |
 *   |          |          |  |Transpose   |  |Transpose |    |
 *   |          |          |  +------+-----+  ++---------+    |
 *   |          |          |         |         |              |
 *   |          |          |         |         |              |
 *   |        +-v----------v---------v-----+   |              |
 *   +-------->                            |<--+              |
 *            |       LSTMCell             |                  |
 *            +--------+------------+------+<-----------------+
 *                     |            |
 *                     |            |
 *                     |            |
 *                     |            |
 *                  +--v-+       +--v-+
 *                  | H0 |       | C0 |
 *                  +----+       +----+
 */
class ov::pass::LSTMCellTfKerasFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMCellTfKerasFusion", "0");
    LSTMCellTfKerasFusion();
};
