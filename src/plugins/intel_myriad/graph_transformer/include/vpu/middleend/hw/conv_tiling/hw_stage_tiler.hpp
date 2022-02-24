// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <vpu/model/base.hpp>
#include <vpu/stage_builder.hpp>
#include <vpu/middleend/hw/tiling.hpp>

namespace vpu {

struct HWConvStageOptions;
struct HWConvStageIO;

// Builds graph which composes tiled analogue of the single stage 'origStage'
class HWConvStageTiler {
public:
    HWConvStageTiler() = delete;
    HWConvStageTiler(const HWConvStageTiler&) = delete;

    HWConvStageTiler(const HWConvStageOptions& stageOptions, const HWConvStageIO& stageIO,
                     const Model& model, const Stage& origStage,
                     const StageBuilder::Ptr& stageBuilder, const HwConvTilingPtr& tiling,
                     bool makeExplicitPoolStage);

    DataVector hwInputTiles;
    std::vector<DimValues> hwInputTilesOffsets;

    DataVector hwWeightsTiles;
    std::vector<DimValues> hwWeightsTilesOffsets;

    DataVector hwOutputTiles;
    std::vector<DimValues> hwOutputTilesOffsets;

    Data hwInput;
    Data hwOutput;

private:
    void addPoolStage(const HWConvStageIO& stageIO, const HWConvStageOptions& stageOptions);
    void expandInput(int numChannels);

    Data createBiases(const HwConvTilingPtr& tiling, const HWConvStageIO& stageIO,
                      const HWConvStageOptions& stageOptions);
    Data createScales(const HwConvTilingPtr& tiling, const HWConvStageIO& stageIO,
                      const HWConvStageOptions& stageOptions);

    Data createInputTile(const HwConvPlaneTilePtr& planeTile, const HwConvChannelTilePtr& channelTile,
                         const std::string& tilePostfix, const HwConvTilingPtr& tiling);

    Data createOutputTile(const HwConvPlaneTilePtr& planeTile, const std::string& tilePostfix,
                          const HwConvTilingPtr& tiling);

    Data reducePartialOverChannelsOutputs(const Data& hwOutputPlaneTile, const HwConvPlaneTilePtr& planeTile,
                                          const HwConvChannelTilePtr& channelTile,
                                          const std::string& channelTilePostfix, const std::string& tilePostfix,
                                          const HWConvStageOptions& stageOptions, const HwConvTilingPtr& tiling,
                                          Data& prevPartialSum);

    Data createTileBiases(const Data& hwBiases, const HwConvChannelTilePtr& channelTile);

    Data createConstTileWeights(const HwConvChannelTilePtr& channelTile, const std::string& channelTilePostfix,
                                const HWConvStageIO& io, const HWConvStageOptions& options);

    Data createIntermediateTileWeights(const HwConvChannelTilePtr& channelTile, const std::string& channelTilePostfix,
                                       const HWConvStageIO& io, const HWConvStageOptions& options);

    void createHWStageForTile(const Data& hwInputTile, const Data& hwTileWeights, const Data& hwTileBiases,
                              const Data& hwScales, const Data& hwOutputTile, bool tileStageWithPool,
                              const HwConvChannelTilePtr& channelTile, const HWConvStageOptions& stageOptions,
                              const std::string& tilePostfix, const HwConvTilingPtr& tiling);

    Model _model;
    StageBuilder::Ptr _stageBuilder;
    Stage _original;
};

struct HWConvStageIO {
public:
    HWConvStageIO() = delete;
    HWConvStageIO(const HWConvStageIO&) = delete;

    HWConvStageIO(const Stage& origStage, const Data& originOutput) {
        origInput = origStage->input(0);
        origWeights = origStage->input(1);
        origBiases = origStage->input(2);
        origScales = origStage->input(3);
        origOutput = originOutput;
        origOutputDesc = origStage->attrs().getOrDefault<DataDesc>("origConvOutput", origOutput->desc());
    }

    Data origInput;
    Data origWeights;
    Data origBiases;
    Data origScales;
    Data origOutput;
    DataDesc origOutputDesc;
};

// Attributes of the stage collected into the structure
struct HWConvStageOptions {
public:
    HWConvStageOptions() = delete;
    HWConvStageOptions(const HWConvStageOptions&) = delete;

    explicit HWConvStageOptions(const Stage& origStage) {
        kernelSizeX = origStage->attrs().get<int>("kernelSizeX");
        kernelSizeY = origStage->attrs().get<int>("kernelSizeY");
        kernelStride = origStage->attrs().get<int>("kernelStrideX");
        padLeft = origStage->attrs().get<int>("padLeft");
        padRight = origStage->attrs().get<int>("padRight");
        padTop = origStage->attrs().get<int>("padTop");
        padBottom = origStage->attrs().get<int>("padBottom");

        withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);
        negativeSlope = origStage->attrs().getOrDefault<float>("negativeSlope", 0.0f);
        a0 = origStage->attrs().getOrDefault<uint32_t>("a0", 0);
        a1 = origStage->attrs().getOrDefault<uint32_t>("a1", 0);
        reluScale = origStage->attrs().getOrDefault<float>("reluScale", 1.0f);

        withClamp = origStage->attrs().getOrDefault<bool>("withClamp", false);
        clampMax = origStage->attrs().getOrDefault<float>("clampMax", 6.0);

        withPool = origStage->attrs().getOrDefault<bool>("withPool", false);
        poolKernelSizeX = origStage->attrs().getOrDefault<int>("poolKernelSizeX", 0);
        poolKernelSizeY = origStage->attrs().getOrDefault<int>("poolKernelSizeY", 0);
        poolKernelStride = origStage->attrs().getOrDefault<int>("poolKernelStride", 0);
        poolPadLeft = origStage->attrs().getOrDefault<int>("poolPadLeft", 0);
        poolPadRight = origStage->attrs().getOrDefault<int>("poolPadRight", 0);
        poolPadTop = origStage->attrs().getOrDefault<int>("poolPadTop", 0);
        poolPadBottom = origStage->attrs().getOrDefault<int>("poolPadBottom", 0);

        scaleFactor = origStage->attrs().getOrDefault<float>("scaleFactor", 1.0f);
    }

    int kernelSizeX;
    int kernelSizeY;
    int kernelStride;
    int padLeft;
    int padRight;
    int padTop;
    int padBottom;

    bool withReLU;
    float negativeSlope;
    uint32_t a0;
    uint32_t a1;
    float reluScale;

    bool withClamp;
    float clampMax;

    bool withPool;
    int poolKernelSizeX;
    int poolKernelSizeY;
    int poolKernelStride;
    int poolPadLeft;
    int poolPadRight;
    int poolPadTop;
    int poolPadBottom;

    float scaleFactor;
};

}  // namespace vpu
