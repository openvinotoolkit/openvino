// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Strips the no-op Reshape + Squeeze the GGUF frontend places between the input ids and the vocabulary Gather, so the
// ids reach the Gather straight from the Parameter as the host gather requires. Needs static shapes to prove the
// Reshape + Squeeze pair is an identity.
class StripNoopGatherIdsReshape : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::StripNoopGatherIdsReshape");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
