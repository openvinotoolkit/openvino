// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf {
class SharedWeights;
}  // namespace ov::frontend::gguf

namespace ov::frontend::gguf::pass {

class MaterializeGGUFWeights : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::MaterializeGGUFWeights");
    // With `shared_weights`, raw cgraph weights are taken from it instead of being extracted again.
    explicit MaterializeGGUFWeights(std::shared_ptr<SharedWeights> shared_weights = {});
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<SharedWeights> m_shared_weights;
};

}  // namespace ov::frontend::gguf::pass
