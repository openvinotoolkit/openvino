// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <vpu/model/base.hpp>
#include <vpu/frontend/stage_builder.hpp>
#include <vpu/hw/tiling.hpp>

namespace vpu {

struct HWConvStageOptions;
struct HWConvStageIO;

// Builds graph which composes tiled analogue of the single stage 'origStage'
class HWConvStageTiler {
private:
    HWConvStageTiler() = delete;
    HWConvStageTiler(const HWConvStageTiler&) = delete;

public:
    DataVector hwInputTiles;
    std::vector<DimValues> hwInputTilesOffsets;

    DataVector hwOutputTiles;
    std::vector<DimValues> hwOutputTilesOffsets;

    Data hwInput;
    Data hwOutput;

    HWConvStageTiler(const HWConvStageOptions &so, const HWConvStageIO &sio,
                 const Model::Ptr &model, const Handle <StageNode> &origStage,
                 const StageBuilder::Ptr &stageBuilder, const HwConvTilingPtr &tiling,
                 const bool makeExplicitPoolStage);
};

struct HWConvStageIO {
private:
    HWConvStageIO() = delete;
    HWConvStageIO(const HWConvStageIO&) = delete;

public:
    Data origInput;
    Data origWeights;
    Data origBiases;
    Data origOutput;
    DataDesc origOutputDesc;

    explicit HWConvStageIO(const Handle<StageNode> &origStage, const Data &originOutput) {
        origInput = origStage->input(0);
        origWeights = origStage->input(1);
        origBiases = origStage->input(2);
        origOutput = originOutput;
        origOutputDesc = origStage->attrs().getOrDefault<DataDesc>("origConvOutput", origOutput->desc());
    }
};

// Attributes of the stage collected into the structure
struct HWConvStageOptions {
private:
    HWConvStageOptions() = delete;
    HWConvStageOptions(const HWConvStageOptions&) = delete;

public:
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

    explicit HWConvStageOptions(const Handle<StageNode> &origStage) {
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
};

}  // namespace vpu
