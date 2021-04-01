// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <algorithm>
#include <set>
#include <memory>

#include <precision_utils.h>

#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(finalizeHwOps);

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW)
            continue;

        HwOpList hwOps;

        auto opType = stage->attrs().get<HwOpType>("hwOpType");

        if (opType == HwOpType::CONV || opType == HwOpType::CONV_POOL) {
            auto input = stage->input(0);
            auto biases = stage->input(2);
            auto scales = stage->input(3);
            auto output = stage->output(0);

            auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
            auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
            auto kernelStride = stage->attrs().get<int>("kernelStride");

            auto pad = stage->attrs().get<HwPaddingInfo>("pad");

            auto tiling = stage->attrs().get<HwConvTileInfo>("tiling");

            auto withReLU = stage->attrs().getOrDefault<bool>("withReLU", false);
            auto a0 = stage->attrs().getOrDefault<uint32_t>("a0", 0);
            auto a1 = stage->attrs().getOrDefault<uint32_t>("a1", 0);

            auto withClamp = stage->attrs().getOrDefault<bool>("withClamp", false);
            auto clampMax = stage->attrs().getOrDefault<float>("clampMax", 6.0);

            auto poolKernelSizeX = stage->attrs().getOrDefault<int>("poolKernelSizeX", 0);
            auto poolKernelSizeY = stage->attrs().getOrDefault<int>("poolKernelSizeY", 0);

            IE_ASSERT(tiling.numDescr > 0);

            int outChanOffset = 0;
            for (int outTileIndex = 0; outTileIndex < tiling.numDescr; ++outTileIndex) {
                auto outNumChans = outTileIndex == tiling.numDescr - 1 ? tiling.lastOutChans : tiling.outChansPerDescr;

                HwOpParams hwOpParams;

                hwOpParams.opType = opType;
                hwOpParams.opMode = tiling.mode;

                if (pad.enable) {
                    hwOpParams.withPad = true;
                    hwOpParams.padMode = HwPadMode::PAD_WITH_ZEROS;
                }

                int bufInd = 0;
                hwOpParams.inputInd = bufInd++;
                hwOpParams.coeffsInd = bufInd++;
                if (biases->usage() != DataUsage::Fake) {
                    hwOpParams.biasesInd = bufInd++;
                }
                if (scales->usage() != DataUsage::Fake) {
                    hwOpParams.scalesInd = bufInd++;
                }
                hwOpParams.outputInd = bufInd++;

                hwOpParams.outChanOffset = outChanOffset;
                hwOpParams.outNumChans = outNumChans;

                hwOpParams.kernelWidth = kernelSizeX;
                hwOpParams.kernelHeight = kernelSizeY;
                hwOpParams.kernelStride = kernelStride;

                if (opType == HwOpType::CONV_POOL) {
                    hwOpParams.poolKernelWidth = poolKernelSizeX;
                    hwOpParams.poolKernelHeight = poolKernelSizeY;
                }

                if (withReLU) {
                    hwOpParams.withReLU = true;
                    hwOpParams.t0 = 0;
                    hwOpParams.a0 = a0;
                    hwOpParams.a1 = a1;
                }
                if (withClamp) {
                    hwOpParams.withClamp = true;
                    hwOpParams.clampMaxVal = clampMax;
                }

                hwOps.vec.emplace_back(hwOpParams);

                outChanOffset += outNumChans;
            }
        } else if (opType == HwOpType::POOL) {
            auto input = stage->input(0);
            auto output = stage->output(0);

            auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
            auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
            auto kernelStride = stage->attrs().get<int>("kernelStride");

            auto poolType = stage->attrs().get<HwPoolType>("poolType");

            auto pad = stage->attrs().get<HwPaddingInfo>("pad");

            auto tiling = stage->attrs().get<HwPoolTileInfo>("tiling");

            auto withReLU = stage->attrs().get<bool>("withReLU");

            auto origDimC = output->desc().dim(Dim::C);
            auto origDimN = output->desc().dim(Dim::N, 1);

            auto hwDimC = origDimN * origDimC;

            IE_ASSERT(tiling.numDescr > 0);

            int chanOffset = 0;
            for (int outTileIndex = 0; outTileIndex < tiling.numDescr; ++outTileIndex) {
                auto numChans =
                    outTileIndex == tiling.numDescr - 1 ?
                        hwDimC - outTileIndex * tiling.chansPerDescr :
                        tiling.chansPerDescr;

                HwOpParams hwOpParams;

                hwOpParams.opType = opType;
                hwOpParams.opMode = tiling.mode;

                hwOpParams.poolType = poolType;

                if (pad.enable) {
                    HwPadMode padType = HwPadMode::PAD_WITH_ZEROS;

                    if (poolType == HwPoolType::MAX) {
                        if (pad.left > 0) {
                            padType = padType | HwPadMode::PAD_REPEAT_LEFT_EDGE;
                        }
                        if (pad.right > 0) {
                            padType = padType | HwPadMode::PAD_REPEAT_RIGHT_EDGE;
                        }
                        if (pad.top > 0) {
                            padType = padType | HwPadMode::PAD_REPEAT_TOP_EDGE;
                        }
                        if (pad.bottom > 0) {
                            padType = padType | HwPadMode::PAD_REPEAT_BOTTOM_EDGE;
                        }
                    }

                    hwOpParams.withPad = true;
                    hwOpParams.padMode = padType;
                }

                int bufInd = 0;
                hwOpParams.inputInd = bufInd++;
                hwOpParams.outputInd = bufInd++;

                hwOpParams.outChanOffset = chanOffset;
                hwOpParams.outNumChans = numChans;

                hwOpParams.kernelWidth = kernelSizeX;
                hwOpParams.kernelHeight = kernelSizeY;
                hwOpParams.kernelStride = kernelStride;

                if (withReLU) {
                    hwOpParams.withReLU = true;
                    hwOpParams.t0 = 0;
                    hwOpParams.a0 = 0;
                    hwOpParams.a1 = 1;
                }

                hwOps.vec.emplace_back(hwOpParams);

                chanOffset += tiling.chansPerDescr;
            }
        } else if (opType == HwOpType::FC) {
            auto input = stage->input(0);
            auto biases = stage->input(2);
            auto scales = stage->input(3);
            auto output = stage->output(0);

            auto tiling = stage->attrs().get<HwFullyConnectedTileInfo>("tiling");

            auto withReLU = stage->attrs().get<bool>("withReLU");

            IE_ASSERT(tiling.numOutTiles > 0);
            IE_ASSERT(tiling.numInSubTiles > 0);

            int outputOffset = 0;
            for (int outTileIndex = 0; outTileIndex < tiling.numOutTiles; ++outTileIndex) {
                int inputOffset = 0;
                for (int subInTileIndex = 0; subInTileIndex < tiling.numInSubTiles; ++subInTileIndex) {
                    auto lastSubTile = (subInTileIndex == tiling.numInSubTiles - 1);
                    auto accum = !lastSubTile;

                    HwOpParams hwOpParams;

                    hwOpParams.opType = opType;
                    hwOpParams.opMode = tiling.mode;

                    int bufInd = 0;
                    hwOpParams.inputInd = bufInd++;
                    hwOpParams.coeffsInd = bufInd++;
                    if (biases->usage() != DataUsage::Fake) {
                        hwOpParams.biasesInd = bufInd++;
                    }
                    if (scales->usage() != DataUsage::Fake) {
                        hwOpParams.scalesInd = bufInd++;
                    }
                    hwOpParams.outputInd = bufInd++;

                    hwOpParams.fcInputOffset = inputOffset;
                    hwOpParams.fcInputNum = tiling.workInN;
                    hwOpParams.fcOutputOffset = outputOffset;
                    hwOpParams.fcOutputNum = tiling.workOutN;
                    hwOpParams.fcAccum = accum;

                    if (lastSubTile && withReLU) {
                        hwOpParams.withReLU = true;
                        hwOpParams.t0 = 0;
                        hwOpParams.a0 = 0;
                        hwOpParams.a1 = 1;
                    }

                    hwOps.vec.emplace_back(hwOpParams);

                    inputOffset += tiling.workInN;
                }

                outputOffset += tiling.workOutN;
            }
        }

        IE_ASSERT(!hwOps.vec.empty());

        stage->attrs().set("hwOps", hwOps);
    }
}

}  // namespace

Pass::Ptr PassManager::finalizeHwOps() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
