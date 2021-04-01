// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/hw/pooling_tiling/hw_stage_tiler.hpp>

#include <precision_utils.h>
#include <tuple>
#include <utility>
#include <memory>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/utils/attributes_map.hpp>

namespace vpu {

namespace {

HwPaddingInfo getPoolPadding(const HwPoolPlaneTilePtr& tile, const DimValues& dims, const HWPoolStageOptions& options) {
    const auto& widthInfo  = tile->widthInfo;
    const auto& heightInfo = tile->heightInfo;

    const auto padW =
        (widthInfo.outputWithJunk  - 1) * options.kernelStride + options.kernelSizeX - widthInfo.inputWithJunk;
    const auto padH =
        (heightInfo.outputWithJunk - 1) * options.kernelStride + options.kernelSizeY - heightInfo.inputWithJunk;

    HwPaddingInfo pad;
    pad.left   = options.padLeft;
    pad.right  = (dims[Dim::W] <= widthInfo.inputEndIndex)  ? options.padRight  : padW - pad.left;
    pad.top    = options.padTop;
    pad.bottom = (dims[Dim::H] <= heightInfo.inputEndIndex) ? options.padBottom : padH - pad.top;

    pad.enable = pad.left || pad.right || pad.top || pad.bottom;

    return pad;
}

}  // namespace

Data HWPoolStageTiler::createInputTile(const HwPoolPlaneTilePtr& planeTile, const HwPoolChannelTilePtr& channelTile,
                                       const std::string& tilePostfix, const HwPoolTilingPtr& tiling) {
    Data hwInputTile;
    if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
        hwInputTile = hwInput;
    } else {
        auto newDesc = hwInput->desc();
        newDesc.setDim(Dim::W, planeTile->widthInfo.inputWithJunk);
        newDesc.setDim(Dim::H, planeTile->heightInfo.inputWithJunk);
        newDesc.setDim(Dim::N, channelTile->numInputChannels);

        hwInputTile = _model->duplicateData(
            hwInput,
            tilePostfix,
            newDesc);

        hwInputTiles.emplace_back(hwInputTile);
        hwInputTilesOffsets.emplace_back(DimValues({
            {Dim::W, planeTile->widthInfo.inputStartIndex},
            {Dim::H, planeTile->heightInfo.inputStartIndex},
            {Dim::N, channelTile->channelStartIndex}}));
    }

    if ((planeTile->widthInfo.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
        const auto hwInputTileAligned = _model->duplicateData(hwInputTile, "@aligned");

        _stageBuilder->addCopyStage(
            _model,
            _original->name() + tilePostfix + "@align-input-ptr",
            _original->origLayer(),
            hwInputTile,
            hwInputTileAligned,
            "HWPoolTiler::input");

        hwInputTile = hwInputTileAligned;
    }

    return hwInputTile;
}

Data HWPoolStageTiler::createOutputTile(const HwPoolPlaneTilePtr& planeTile, const HwPoolChannelTilePtr& channelTile,
                                        const std::string& tilePostfix, const HwPoolTilingPtr& tiling) {
    Data hwOutputTile;
    if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
        hwOutputTile = hwOutput;
    } else {
        auto newDesc = hwOutput->desc();
        newDesc.setDim(Dim::W, planeTile->widthInfo.outputEndIndex - planeTile->widthInfo.outputStartIndex);
        newDesc.setDim(Dim::H, planeTile->heightInfo.outputEndIndex - planeTile->heightInfo.outputStartIndex);
        newDesc.setDim(Dim::N, channelTile->numInputChannels);

        hwOutputTile = _model->duplicateData(hwOutput, tilePostfix, newDesc);

        hwOutputTiles.emplace_back(hwOutputTile);
        hwOutputTilesOffsets.emplace_back(DimValues({
            {Dim::W, planeTile->widthInfo.outputStartIndex},
            {Dim::H, planeTile->heightInfo.outputStartIndex},
            {Dim::N, channelTile->channelStartIndex}}));
    }


    if ((planeTile->widthInfo.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
        auto hwOutputTileAligned = _model->duplicateData(hwOutputTile, "@aligned");

        _stageBuilder->addCopyStage(
            _model,
            _original->name() + tilePostfix + "@align-output-ptr",
            _original->origLayer(),
            hwOutputTileAligned,
            hwOutputTile,
            "HWPoolTiler::output");

        hwOutputTile = hwOutputTileAligned;
    }

    if (planeTile->heightInfo.outputJunkBefore != 0 || planeTile->heightInfo.outputJunkAfter != 0 ||
        planeTile->widthInfo.outputJunkBefore != 0 || planeTile->widthInfo.outputJunkAfter != 0) {
        auto newDesc = hwOutputTile->desc();
        newDesc.setDim(Dim::W, planeTile->widthInfo.outputWithJunk);
        newDesc.setDim(Dim::H, planeTile->heightInfo.outputWithJunk);

        auto hwOutputTileWithJunk = _model->duplicateData(hwOutputTile, "@with-junk", newDesc);

        DimValues innerOffset;
        innerOffset.set(Dim::W, planeTile->widthInfo.outputJunkBefore);
        innerOffset.set(Dim::H, planeTile->heightInfo.outputJunkBefore);

        _stageBuilder->addCropStage(
            _model,
            _original->name() + tilePostfix + "@remove-junk",
            _original->origLayer(),
            hwOutputTileWithJunk,
            hwOutputTile,
            innerOffset);

        hwOutputTile = hwOutputTileWithJunk;
    }

    return hwOutputTile;
}

void HWPoolStageTiler::createHWStageForTile(const Data& hwInputTile,
                                            const Data& hwOutputTile,
                                            const HwPoolPlaneTilePtr& planeTile,
                                            const HwPoolChannelTilePtr& channelTile,
                                            const HWPoolStageOptions& stageOptions,
                                            const std::string& tilePostfix,
                                            const HwPoolTilingPtr& tiling) {
    auto hwTileWeights = _model->addFakeData();
    auto hwTileBiases = _model->addFakeData();
    auto hwTileScales = _model->addFakeData();

    auto hwStage = _model->addNewStage<MyriadXHwStage>(
        _original->name() + tilePostfix,
        StageType::MyriadXHwOp,
        _original->origLayer(),
        {hwInputTile, hwTileWeights, hwTileBiases, hwTileScales},
        {hwOutputTile});

    hwStage->attrs().set<HwOpType>("hwOpType", HwOpType::POOL);
    const auto& type = _original->type() == StageType::StubMaxPool ? HwPoolType::MAX : HwPoolType::AVERAGE;
    hwStage->attrs().set<HwPoolType>("poolType", type);

    hwStage->attrs().set<int>("kernelSizeX", stageOptions.kernelSizeX);
    hwStage->attrs().set<int>("kernelSizeY", stageOptions.kernelSizeY);
    hwStage->attrs().set<int>("kernelStride", stageOptions.kernelStride);

    hwStage->attrs().set("pad", getPoolPadding(planeTile, hwInput->desc().dims(), stageOptions));

    hwStage->attrs().set<HwPoolTileInfo>("tiling", channelTile->finalTiles);

    hwStage->attrs().set<bool>("withReLU", stageOptions.withReLU);
}

HWPoolStageTiler::HWPoolStageTiler(const HWPoolStageOptions& stageOptions, const HWPoolStageIO& stageIO,
                                   const Model& model, const Stage& origStage,
                                   const StageBuilder::Ptr& stageBuilder, const HwPoolTilingPtr& tiling) {
    hwInput = stageIO.origInput;
    hwOutput = stageIO.origOutput;

    _model = model;
    _stageBuilder = stageBuilder;
    _original = origStage;

    hwInputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwInputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);

    for (const auto& planeTile : tiling->planeTiles) {
        for (const auto& channelTile : planeTile->channelTiles) {
            const auto tilePostfix = getPlaneTilePostfix(planeTile) + getChannelTilePostfix(channelTile);

            const auto hwInputTile = createInputTile(planeTile, channelTile, tilePostfix, tiling);
            const auto hwOutputTile = createOutputTile(planeTile, channelTile, tilePostfix, tiling);

            createHWStageForTile(hwInputTile, hwOutputTile, planeTile, channelTile, stageOptions, tilePostfix, tiling);
        }
    }
}
}  // namespace vpu
