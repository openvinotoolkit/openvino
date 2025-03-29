// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

/*
 * Description:
 *     ConvertFqRnnToQuantizedRnn detects RNN / LSTM / GRU_RNN operations
 *     with FQ operations on the inputs and forms a new TypeRelaxed operation
 *     with quantization parameters as runtime parameters of the operation.
 *     @todo add ascii graph examples
 *
 * Before:
 *
 * +-------+   +-------+   +-------+  +-------+  +-------+  +-------+
 * |   X   |   |   H   |   |   C   |  |   W   |  |   R   |  |   B   |
 * |       |   |       |   |       |  |       |  |       |  |       |
 * | u8/i8 |   | u8/i8 |   |  f32  |  |  i8   |  |  i8   |  |  f32  |
 * +---+---+   +---+---+   +---+---+  +---+---+  +---+---+  +---+---+
 *     |           |           |          |          |          |
 * +---v---+   +---v---+       |      +---v---+  +---v---+      |
 * |       |   |       |       |      |       |  |       |      |
 * |  deq  |   |  deq  |       |      |  deq  |  |  deq  |      |
 * |       |   |       |       |      |       |  |       |      |
 * +---+---+   +---+---+       |      +---+---+  +---+---+      |
 *     |           |           |          |          |          |
 *     |           |           |          |          |          |
 * +---v-----------v-----------v----------v----------v----------v---+
 * |                                                                |
 * |            LSTMSequence / GRUSequence (f32)                    |
 * |                                                                |
 * +---------------+-----------+----------+-------------------------+
 *                 |           |          |
 *                 |Y f32      |Ho f32    |Co f32
 *                 |           |          |
 *                 |           |          |
 *                 |           |          |
 *                 v           v          v
 *
 *                                                                               v
 *
 *
 * After:
 *
 * +-------+   +-------+   +-------+  +-------+  +-------+  +-------+
 * |   X   |   |   H   |   |   C   |  |   W   |  |   R   |  |   B   |
 * |       |   |       |   |       |  |       |  |       |  |       |
 * | u8/i8 |   | u8/i8 |   |  f32  |  |  i8   |  |  i8   |  |  f32  |
 * +---+---+   +---+---+   +---+---+  +---+---+  +---+---+  +---+---+
 *     |           |           |          |          |          |
 *     |           |           |          |          |          |
 * +---v-----------v-----------v----------v----------v----------v---+
 * |         TypeRelaxed                  rt_info[inputScales]      |
 * |                                                                |
 * |  LSTMSequence / GRUSequence (u8/i8)  rt_into[weightsScales]    |
 * +---------------+-----------+----------+-------------------------+
 *                 |           |          |
 *                 |Y f32      |Ho u8/i8  |Co f32
 *                 |           |          |
 *                 |       +---v---+      |
 *                 |       |       |      |
 *                 |       |  deq  |      |
 *                 |       |       |      |
 *                 |       +---+---+      |
 *                 |           |          |
 *                 |           |          |
 *                 |           |          |
 *                 v           v          v
 */

namespace ov::intel_cpu {

class ConvertFqRnnToQuantizedRnn : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFqRnnToQuantizedRnn");
    ConvertFqRnnToQuantizedRnn();
};

}  // namespace ov::intel_cpu
