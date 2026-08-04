// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API RemoveConcatSliceAfterLoopSelectiveSSM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConcatSliceAfterLoopSelectiveSSM");
    RemoveConcatSliceAfterLoopSelectiveSSM();
};

class TRANSFORMATIONS_API FuseSelectiveSSMLoop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseSelectiveSSMLoop");
    FuseSelectiveSSMLoop();
};

class TRANSFORMATIONS_API SelectiveSSMFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SelectiveSSMFusion");
    SelectiveSSMFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
