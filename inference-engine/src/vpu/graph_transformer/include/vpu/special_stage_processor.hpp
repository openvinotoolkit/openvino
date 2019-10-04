// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <vpu/model/stage.hpp>
#include <vpu/model/model.hpp>
#include <vpu/frontend/stage_builder.hpp>

namespace vpu {

class SpecialStageProcessor final {
public:
    inline explicit SpecialStageProcessor(const StageBuilder::Ptr& stageBuilder) :
            _stageBuilder(stageBuilder) {
    }

    void processSplit(
            const Model::Ptr& model,
            const Stage& stage);

    void processConcat(
            const Model::Ptr& model,
            const Stage& stage);

    void processReshape(
            const Model::Ptr& model,
            const Stage& stage);

    void processExpand(
            const Model::Ptr& model,
            const Stage& stage);

    void processShrink(
            const Model::Ptr& model,
            const Stage& stage);

private:
    StageBuilder::Ptr _stageBuilder;
};

}  // namespace vpu
