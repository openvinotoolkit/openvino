// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_pass: graph-level optimization passes over the plugin-agnostic IR.
// Every pass is a pure function ir_graph -> ir_graph; they never touch
// runtime state, so they are trivially testable on CPU.
//
//   dce             — drop nodes unreachable from the model outputs
//                     (dead branch elimination)
//   fold_constants  — evaluate every node whose inputs are all constants,
//                     replacing it with a new constant; iterated to fixpoint
//   peephole        — local rewrites: transpose∘transpose -> composed
//                     transpose (identity cancels), relu∘relu -> relu,
//                     sigmoid∘sigmoid -> sigmoid
//   optimize        — dce + fold + peephole until nothing changes

#pragma once

#include "vk_ir.hpp"

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace pass {

[[nodiscard]] ir_graph dce(const ir_graph& g);
[[nodiscard]] ir_graph fold_constants(const ir_graph& g);
[[nodiscard]] ir_graph peephole(const ir_graph& g);
[[nodiscard]] ir_graph optimize(const ir_graph& g);

}  // namespace pass
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov
