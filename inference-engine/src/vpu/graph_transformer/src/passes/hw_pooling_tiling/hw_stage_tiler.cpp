// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <vpu/passes/hw_pooling_tiling/hw_stage_tiler.hpp>

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
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/utils/attributes_map.hpp>

namespace vpu {

HwPaddingInfo getPoolPadding(const HwPlaneTilePtr<HwPoolTileInfo>& tile,
                             const DimValues& dims,
                             int kernelSizeX,
                             int kernelSizeY,
                             int kernelStrideX,
                             int kernelStrideY,
                             int padLeft,
                             int padRight,
                             int padTop,
                             int padBottom) {
    const auto& widthInfo  = tile->widthInfo;
    const auto& heightInfo = tile->heightInfo;

    auto padW = (widthInfo.outputWithJunk  - 1)*kernelStrideX + kernelSizeX - widthInfo.inputWithJunk;
    auto padH = (heightInfo.outputWithJunk - 1)*kernelStrideY + kernelSizeY - heightInfo.inputWithJunk;

    HwPaddingInfo pad;

    pad.left   = padLeft;
    pad.right  = (dims[Dim::W] <= widthInfo.inputEndIndex)  ? padRight  : padW - pad.left;
    pad.top    = padTop;
    pad.bottom = (dims[Dim::H] <= heightInfo.inputEndIndex) ? padBottom : padH - pad.top;

    pad.enable = pad.left || pad.right || pad.top || pad.bottom;

    return pad;
}

HWPoolStageTiler::HWPoolStageTiler(const HWPoolStageOptions &so, const HWPoolStageIO &sio,
             const Model::Ptr &model, const Handle <StageNode> &origStage,
             const StageBuilder::Ptr &stageBuilder, const HwPoolTilingPtr &tiling) {
    hwInput = sio.origInput;
    hwOutput = sio.origOutput;

    hwInputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwInputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTiles.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);
    hwOutputTilesOffsets.reserve(tiling->socTiles * tiling->sohTiles * tiling->sowTiles);

    for (const auto& planeTile : tiling->planeTiles) {
        for (const auto& channelTile : planeTile->channelTiles) {
            auto tilePostfix = getPlaneTilePostfix(planeTile) + getChannelTilePostfix(channelTile);

            //
            // Create input tile
            //

            Data hwInputTile;

            if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
                hwInputTile = hwInput;
            } else {
                auto newDesc = hwInput->desc();
                newDesc.setDim(Dim::W, planeTile->widthInfo.inputWithJunk);
                newDesc.setDim(Dim::H, planeTile->heightInfo.inputWithJunk);
                newDesc.setDim(Dim::N, channelTile->numInputChannels);

                hwInputTile = model->duplicateData(
                        hwInput,
                        tilePostfix,
                        newDesc);

                hwInputTiles.emplace_back(hwInputTile);
                hwInputTilesOffsets.emplace_back(
                        DimValues({
                                          {Dim::W, planeTile->widthInfo.inputStartIndex},
                                          {Dim::H, planeTile->heightInfo.inputStartIndex},
                                          {Dim::N, channelTile->channelStartIndex}
                                  }));
            }

            //
            // Add alignement to input tile if needed
            //

            if ((planeTile->widthInfo.inputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                auto hwInputTileAligned = model->duplicateData(
                        hwInputTile,
                        "@aligned");

                stageBuilder->addCopyStage(
                        model,
                        origStage->name() + tilePostfix + "@align-input-ptr",
                        origStage->origLayer(),
                        hwInputTile,
                        hwInputTileAligned);

                hwInputTile = hwInputTileAligned;
            }

            //
            // Create output tile
            //

            Data hwOutputTile;

            if (tiling->sohTiles == 1 && tiling->sowTiles == 1 && tiling->socTiles == 1) {
                hwOutputTile = hwOutput;
            } else {
                auto newDesc = hwOutput->desc();
                newDesc.setDim(Dim::W, planeTile->widthInfo.outputEndIndex - planeTile->widthInfo.outputStartIndex);
                newDesc.setDim(Dim::H, planeTile->heightInfo.outputEndIndex - planeTile->heightInfo.outputStartIndex);
                newDesc.setDim(Dim::N, channelTile->numInputChannels);

                hwOutputTile = model->duplicateData(
                        hwOutput,
                        tilePostfix,
                        newDesc);

                hwOutputTiles.emplace_back(hwOutputTile);
                hwOutputTilesOffsets.emplace_back(
                        DimValues({
                                          {Dim::W, planeTile->widthInfo.outputStartIndex},
                                          {Dim::H, planeTile->heightInfo.outputStartIndex},
                                          {Dim::N, channelTile->channelStartIndex}
                                  }));
            }

            //
            // Add alignement to output tile if needed
            //

            if ((planeTile->widthInfo.outputStartIndex * sizeof(fp16_t)) % 16 != 0) {
                auto hwOutputTileAligned = model->duplicateData(
                        hwOutputTile,
                        "@aligned");

                stageBuilder->addCopyStage(
                        model,
                        origStage->name() + tilePostfix + "@align-output-ptr",
                        origStage->origLayer(),
                        hwOutputTileAligned,
                        hwOutputTile);

                hwOutputTile = hwOutputTileAligned;
            }

            //
            // Process output junk if needed
            //

            if (planeTile->heightInfo.outputJunkBefore != 0 ||
                planeTile->heightInfo.outputJunkAfter != 0 ||
                planeTile->widthInfo.outputJunkBefore != 0 ||
                planeTile->widthInfo.outputJunkAfter != 0) {
                auto newDesc = hwOutputTile->desc();
                newDesc.setDim(Dim::W, planeTile->widthInfo.outputWithJunk);
                newDesc.setDim(Dim::H, planeTile->heightInfo.outputWithJunk);

                auto hwOutputTileWithJunk = model->duplicateData(
                        hwOutputTile,
                        "@with-junk",
                        newDesc);

                DimValues innerOffset;
                innerOffset.set(Dim::W, planeTile->widthInfo.outputJunkBefore);
                innerOffset.set(Dim::H, planeTile->heightInfo.outputJunkBefore);

                stageBuilder->addShrinkStage(
                        model,
                        origStage->name() + tilePostfix + "@remove-junk",
                        origStage->origLayer(),
                        hwOutputTileWithJunk,
                        hwOutputTile,
                        innerOffset);

                hwOutputTile = hwOutputTileWithJunk;
            }

            //
            // Create HW stage for tile
            //

            auto hwPad = getPoolPadding(
                    planeTile, hwInput->desc().dims(),
                    so.kernelSizeX, so.kernelSizeY,
                    so.kernelStride, so.kernelStride,
                    so.padLeft, so.padRight, so.padTop, so.padBottom);

            auto hwTileWeights = model->addFakeData();
            auto hwTileBiases = model->addFakeData();
            auto hwTileScales = model->addFakeData();

            auto hwStage = model->addNewStage<MyriadXHwStage>(
                    origStage->name() + tilePostfix,
                    StageType::MyriadXHwOp,
                    origStage->origLayer(),
                    {hwInputTile, hwTileWeights, hwTileBiases, hwTileScales},
                    {hwOutputTile});

            hwStage->attrs().set<HwOpType>("hwOpType", HwOpType::POOL);
            hwStage->attrs().set<HwPoolType>("poolType", origStage->type() == StageType::StubMaxPool ? HwPoolType::MAX : HwPoolType::AVERAGE);

            hwStage->attrs().set<int>("kernelSizeX", so.kernelSizeX);
            hwStage->attrs().set<int>("kernelSizeY", so.kernelSizeY);
            hwStage->attrs().set<int>("kernelStride", so.kernelStride);

            hwStage->attrs().set("pad", hwPad);

            hwStage->attrs().set<HwPoolTileInfo>("tiling", channelTile->finalTiles);

            hwStage->attrs().set<bool>("withReLU", so.withReLU);
        }
    }
}
}  // namespace vpu
