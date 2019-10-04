// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <vpu/model/base.hpp>
#include <vpu/frontend/stage_builder.hpp>
#include <vpu/hw/tiling.hpp>

namespace vpu {

struct HWPoolStageOptions;
struct HWPoolStageIO;

// Builds graph which composes tiled analogue of the single stage 'origStage'
class HWPoolStageTiler {
private:
    HWPoolStageTiler() = delete;
    HWPoolStageTiler(const HWPoolStageTiler&) = delete;

public:
    DataVector hwInputTiles;
    std::vector<DimValues> hwInputTilesOffsets;

    DataVector hwOutputTiles;
    std::vector<DimValues> hwOutputTilesOffsets;

    Data hwInput;
    Data hwOutput;

    HWPoolStageTiler(const HWPoolStageOptions &so, const HWPoolStageIO &sio,
                 const Model::Ptr &model, const Handle <StageNode> &origStage,
                 const StageBuilder::Ptr &stageBuilder, const HwPoolTilingPtr &tiling);
};

struct HWPoolStageIO {
private:
    HWPoolStageIO() = delete;
    HWPoolStageIO(const HWPoolStageIO&) = delete;

public:
    Data origInput;
    Data origOutput;

    explicit HWPoolStageIO(const Handle<StageNode> &origStage, const Data &originOutput) {
        origInput = origStage->input(0);
        origOutput = originOutput;
    }
};

// Attributes of the stage collected into the structure
struct HWPoolStageOptions {
private:
    HWPoolStageOptions() = delete;
    HWPoolStageOptions(const HWPoolStageOptions&) = delete;

public:
    int kernelSizeX;
    int kernelSizeY;
    int kernelStride;
    int padLeft;
    int padRight;
    int padTop;
    int padBottom;

    bool withReLU;

    explicit HWPoolStageOptions(const Handle<StageNode> &origStage) {
        kernelSizeX = origStage->attrs().get<int>("kernelSizeX");
        kernelSizeY = origStage->attrs().get<int>("kernelSizeY");
        kernelStride = origStage->attrs().get<int>("kernelStrideX");
        padLeft = origStage->attrs().get<int>("padLeft");
        padRight = origStage->attrs().get<int>("padRight");
        padTop = origStage->attrs().get<int>("padTop");
        padBottom = origStage->attrs().get<int>("padBottom");

        withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);
    }
};

}  // namespace vpu
