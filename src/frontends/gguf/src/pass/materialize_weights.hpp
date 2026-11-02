// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

class MaterializeGGUFWeights : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::MaterializeGGUFWeights");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::frontend::gguf::pass
