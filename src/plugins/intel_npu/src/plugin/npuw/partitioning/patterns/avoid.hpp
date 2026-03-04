// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace online {
class Snapshot;  // Forward declaration
}  // namespace online

namespace patterns {
namespace avoid {

// Note: this pattern is only utilized by the online partitionerls

class RMSNorm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::RMSNorm");
    RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

class SinCos : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::SinCos");
    SinCos(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

class GemmaRoPE : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::GemmaRoPE");
    GemmaRoPE(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

// Pattern: Interpolate in downsampling case (input spatial dims > output spatial dims)
// Matches any Interpolate node and checks at callback time whether it is downsampling.
class DownsampleInterpolate : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::DownsampleInterpolate");
    DownsampleInterpolate(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                          const std::string& avoid_device);
};

// Pattern: FloorMod and its direct input producer — both need FP32 precision.
class FloorModFP32 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::FloorModFP32");
    FloorModFP32(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

// Pattern: CumSum → Multiply → Transpose → Multiply → Interpolate → Transpose → Sin
// From Kokoro-82M l_sin_gen (sinusoidal position encoding generator).
// CumSum accumulates phase values that can exceed FP16 max (65504), corrupting
// the downstream Sin output.  The entire chain must run in FP32 or on CPU.
class CumSumSinGen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::CumSumSinGen");
    CumSumSinGen(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

// Pattern: RandomUniform → Log → Multiply → Sqrt + RandomUniform → Cos → Multiply
// Box-Muller transform for generating normally-distributed noise from uniform
// random samples.  Log of very small uniform values underflows in FP16.
class BoxMullerNoise : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::BoxMullerNoise");
    BoxMullerNoise(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

// Pattern: Divide → Atan → cascaded Select chain (aten::angle decomposition)
class AngleComplex : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::avoid::AngleComplex");
    AngleComplex(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device);
};

}  // namespace avoid
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
