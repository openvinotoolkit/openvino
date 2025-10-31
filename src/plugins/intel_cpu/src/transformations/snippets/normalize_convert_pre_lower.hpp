// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Android-only integration
#if defined(ANDROID) || defined(__ANDROID__)

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
namespace pass {

// CPU-side pass: for every ov::snippets::op::Subgraph, replace any remaining
// ov::op::v0::Convert / opset1::Convert inside subgraph body with
// ov::snippets::op::ConvertTruncation right before lowering in CPU pipeline.
class NormalizeConvertPreLower final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CPU::NormalizeConvertPreLower");
    NormalizeConvertPreLower();
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov

#endif  // defined(ANDROID) || defined(__ANDROID__)
