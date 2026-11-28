// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace util {

// true if any SDPA in the model has a sink input (6th input, e.g. GPT-OSS)
bool has_sdpa_with_sink(const std::shared_ptr<ov::Model>& model);

// Decompose SDPA into MatMul -> [Add(mask)] -> Softmax -> MatMul. If any SDPA has a sink, the core decomposition is
// used for the whole model. use_high_precision_on_add keeps the mask Add in f32.
class DecomposeSDPA : public ov::pass::ModelPass {
    bool m_use_high_precision_on_add;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DecomposeSDPA");
    explicit DecomposeSDPA(bool use_high_precision_on_add) : m_use_high_precision_on_add(use_high_precision_on_add) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Transpose V tensors, including past/present KV-cache ports. The V consumer is either the decomposed SDPA's MatMul
// (transpose_b is set) or a 4/5-input SDPA (an explicit Transpose before SDPA.V restores the layout it expects).
// is_whisper enables the cross-attention (direct V input) matchers, which match only the decomposed SDPA.
// Direct ShapeOf consumers use a Gather to restore dimension order without freezing dynamic shapes.
// Other consumers are not rewired by the ShapeOf correction.
class OptimizeValueTensors : public ov::pass::ModelPass {
    bool m_is_prefill;
    bool m_is_whisper;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::OptimizeValueTensors");
    explicit OptimizeValueTensors(bool is_prefill, bool is_whisper = false)
        : m_is_prefill(is_prefill),
          m_is_whisper(is_whisper) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace util
}  // namespace npuw
}  // namespace ov
