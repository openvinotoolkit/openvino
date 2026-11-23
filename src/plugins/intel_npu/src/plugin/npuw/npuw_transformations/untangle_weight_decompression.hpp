// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// A weight that several ops read after its decompression (the Convert, the
// zero point Subtract, the scale Multiply) is one tensor they all share. When
// the readers land in different subgraphs, as the embedding Gather and the LM
// head of a model with tied embeddings do, that tensor becomes a link between
// the subgraphs: one produces the whole decompressed weight and hands it to
// the other every inference.
//
// Gives every reader its own copy of the decompression ops, so a decompressed
// weight is never shared. The weight Constants stay shared. It does not depend
// on the weight format: any ops computed from Constants alone, a weight among
// them, are untangled. patterns::opt::untangleWeightConverts() does the same
// for a Convert that reads a Constant directly.
class UntangleWeightDecompression : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::UntangleWeightDecompression");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
