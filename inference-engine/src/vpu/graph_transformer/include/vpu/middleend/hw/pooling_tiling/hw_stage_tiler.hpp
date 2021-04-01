// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <vpu/model/base.hpp>
#include <vpu/stage_builder.hpp>
#include <vpu/middleend/hw/tiling.hpp>

namespace vpu {

struct HWPoolStageOptions;
struct HWPoolStageIO;

// Builds graph which composes tiled analogue of the single stage 'origStage'
class HWPoolStageTiler {
public:
    HWPoolStageTiler() = delete;
    HWPoolStageTiler(const HWPoolStageTiler&) = delete;

    HWPoolStageTiler(const HWPoolStageOptions& stageOptions, const HWPoolStageIO& stageIO,
                     const Model& model, const Stage& origStage,
                     const StageBuilder::Ptr& stageBuilder, const HwPoolTilingPtr& tiling);

    DataVector hwInputTiles;
    std::vector<DimValues> hwInputTilesOffsets;

    DataVector hwOutputTiles;
    std::vector<DimValues> hwOutputTilesOffsets;

    Data hwInput;
    Data hwOutput;

private:
    Data createInputTile(const HwPoolPlaneTilePtr& planeTile, const HwPoolChannelTilePtr& channelTile,
                         const std::string& tilePostfix, const HwPoolTilingPtr& tiling);
    Data createOutputTile(const HwPoolPlaneTilePtr& planeTile, const HwPoolChannelTilePtr& channelTile,
                          const std::string& tilePostfix, const HwPoolTilingPtr& tiling);

    void createHWStageForTile(const Data& hwInputTile, const Data& hwOutputTile, const HwPoolPlaneTilePtr& planeTile,
                              const HwPoolChannelTilePtr& channelTile, const HWPoolStageOptions& stageOptions,
                              const std::string& tilePostfix, const HwPoolTilingPtr& tiling);

    Model _model;
    StageBuilder::Ptr _stageBuilder;
    Stage _original;
};

struct HWPoolStageIO {
public:
    HWPoolStageIO() = delete;
    HWPoolStageIO(const HWPoolStageIO&) = delete;

    explicit HWPoolStageIO(const Stage& origStage, const Data& originOutput) {
        origInput = origStage->input(0);
        origOutput = originOutput;
    }

    Data origInput;
    Data origOutput;
};

// Attributes of the stage collected into the structure
struct HWPoolStageOptions {
public:
    HWPoolStageOptions() = delete;
    HWPoolStageOptions(const HWPoolStageOptions&) = delete;

    explicit HWPoolStageOptions(const Stage& origStage) {
        kernelSizeX = origStage->attrs().get<int>("kernelSizeX");
        kernelSizeY = origStage->attrs().get<int>("kernelSizeY");
        kernelStride = origStage->attrs().get<int>("kernelStrideX");
        padLeft = origStage->attrs().get<int>("padLeft");
        padRight = origStage->attrs().get<int>("padRight");
        padTop = origStage->attrs().get<int>("padTop");
        padBottom = origStage->attrs().get<int>("padBottom");

        withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);
    }

    int kernelSizeX;
    int kernelSizeY;
    int kernelStride;
    int padLeft;
    int padRight;
    int padTop;
    int padBottom;

    bool withReLU;
};

}  // namespace vpu
