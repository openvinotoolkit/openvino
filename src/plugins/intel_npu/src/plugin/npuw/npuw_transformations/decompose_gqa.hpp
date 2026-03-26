// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class DecomposeGQA : public ov::pass::ModelPass {
    bool m_is_prefill_model;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DecomposeGQA");
    explicit DecomposeGQA(bool is_prefill_model);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
