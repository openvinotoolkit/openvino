// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace util {

// SDPA-unroll and transpose transformations
class OptimizeValueTensors : public ov::pass::ModelPass {
    bool m_is_prefill;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::OptimizeValueTensors");
    explicit OptimizeValueTensors(bool is_prefill) : m_is_prefill(is_prefill) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace util
}  // namespace npuw
}  // namespace ov
