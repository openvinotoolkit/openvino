// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// Strips the no-op Reshape + Squeeze the GGUF frontend places between the input ids and the vocabulary Gather, so the
// ids reach the Gather straight from the Parameter as the host gather requires. Needs static shapes to prove the
// Reshape + Squeeze pair is an identity.
class StripNoopGatherIdsReshape : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::StripNoopGatherIdsReshape");
    StripNoopGatherIdsReshape();
};

}  // namespace ov::npuw
