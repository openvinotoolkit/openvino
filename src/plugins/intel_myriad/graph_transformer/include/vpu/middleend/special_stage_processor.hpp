// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <vpu/model/stage.hpp>
#include <vpu/model/model.hpp>
#include <vpu/stage_builder.hpp>

namespace vpu {

class SpecialStageProcessor final {
public:
    inline explicit SpecialStageProcessor(const StageBuilder::Ptr& stageBuilder) :
            _stageBuilder(stageBuilder) {
    }

    void processSplit(
            const Model& model,
            const Stage& stage);

    void processConcat(
            const Model& model,
            const Stage& stage);

    void processReshape(
            const Model& model,
            const Stage& stage);

    void processExpand(
            const Model& model,
            const Stage& stage);

    void processCrop(
            const Model& model,
            const Stage& stage);

    void processLoopStart(const Model& model, const Stage& stage);
    void processLoopEnd(const Model& model, const Stage& stage);

private:
    StageBuilder::Ptr _stageBuilder;
};

}  // namespace vpu
