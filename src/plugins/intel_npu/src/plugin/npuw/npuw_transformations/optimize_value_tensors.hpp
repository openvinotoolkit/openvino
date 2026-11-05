// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace util {

// Decompose SDPA and transpose V tensors, including past/present KV-cache ports.
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
