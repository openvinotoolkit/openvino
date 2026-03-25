// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::npuw {

void run_kv_cache_dynamic_qantization_passes(const std::shared_ptr<ov::Model>& model, ov::element::Type kv_cache_precision_hint);

/// Decomposition passes for ov::op::internal::DynamicQuantize.

/// V1: handcrafted symmetric-style, i8 [-127, 127]
class DecomposeDynamicQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::DecomposeDynamicQuantize");
    DecomposeDynamicQuantize();
};

/// V2: ONNX DynamicQuantizeLinear style, u8 [0, 255]
class DecomposeDynamicQuantize2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::DecomposeDynamicQuantize2");
    DecomposeDynamicQuantize2();
};

/// V3: compiler pattern style, i8 [-128, 127]
class DecomposeDynamicQuantize3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::DecomposeDynamicQuantize3");
    DecomposeDynamicQuantize3();
};

// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw
