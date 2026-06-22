// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Ensures every shared tiny floating-point Constant (typically a DequantizeLinear
// scale) consumed by more than one op gets its own private copy.  Some model
// exporters reuse a single scale node across several layers; this shared ownership
// prevents NPUW's FOLD pass from building a complete per-instance scalar bank,
// causing partitioning to fail.
class UntangleDQScale : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::UntangleDQScale");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
