// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Moves identical per-tensor FakeQuantize ops from every Split/VariadicSplit output to the split input.
 *
 * Models exported with `chunk` as two Slices (later grouped into a VariadicSplit) are quantized with one
 * FakeQuantize per slice instead of one before the split. The producer of the split input then cannot fuse
 * the quantization, so it writes high-precision data, the split runs in high precision and every slice runs
 * a standalone quantize kernel. FakeQuantize is element-wise, so with the same scalar ranges it commutes with
 * the split and a single FakeQuantize on the split input gives bit-identical outputs.
 *
 * Before:                                   After:
 *
 *             Data                                  Data
 *               │                                     │
 *      ┌────────┴────────┐                     ┌──────┴───────┐
 *      │ (Variadic)Split │                     │ FakeQuantize │
 *      └──┬───────────┬──┘                     └──────┬───────┘
 *  ┌──────┴─────┐ ┌───┴──────────┐          ┌────────┴────────┐
 *  │FakeQuantize│ │ FakeQuantize │          │ (Variadic)Split │
 *  └──────┬─────┘ └───┬──────────┘          └──┬───────────┬──┘
 *         │           │                        │           │
 *
 * Applied only when:
 *  - every split output has exactly one consumer and it is the data input of a FakeQuantize, otherwise a
 *    non-quantized consumer would observe quantized data;
 *  - all FakeQuantize ops have the same levels, auto-broadcast and constant ranges, and the ranges are single
 *    elements (per-tensor); per-channel ranges along the split axis would need re-concatenation;
 *  - no FakeQuantize feeds a Result: a multi-output split cannot take over the Result-facing friendly name.
 *
 * A ModelPass because the rewrite rewires the consumers of the FakeQuantize ops, which follow the split.
 */
class MoveFakeQuantizeBeforeSplit : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MoveFakeQuantizeBeforeSplit");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
