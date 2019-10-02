// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <precision_utils.h>
#include <utility>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/passes/hw_conv_tiling/hw_convolution_tiler.hpp>
#include <vpu/passes/hw_conv_tiling/hw_stage_tiler.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(hwConvTiling);

    for (const auto& origStage : model->getStages()) {
        if (origStage->type() != StageType::StubConv) {
            continue;
        }

        const auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        const HWConvStageOptions so(origStage);
        const HWConvStageIO sio(origStage, origStage->output(0));

        //
        // Unsupported paddings
        //

        //
        // Try to find "best" tiling
        //

        const size_t tilingsCount = 1;
        const HWTilingNS::Direction direction =
                HWTilingNS::Direction::INPUT_TO_OUTPUT;
                // HWTilingNS::Direction::OUTPUT_TO_INPUT;

        const HWTilingNS::HWConvolutionTiler tiler1stAttempt(
                HWTilingNS::ConvolutionOptions(origStage->name(),
                     sio.origInput->desc().dims(), sio.origOutput->desc().dims(),
                     sio.origOutputDesc.dims(),
                     so.kernelSizeX, so.kernelSizeY,
                     so.kernelStride,
                     so.padLeft, so.padRight, so.padTop, so.padBottom, so.withPool),
                direction, tilingsCount);

        const HWTilingNS::HWConvolutionTiler& tiler =
                (!tiler1stAttempt.isTilingPossible() && tiler1stAttempt.withPool()) ?
                HWTilingNS::HWConvolutionTiler(
                        HWTilingNS::ConvolutionOptions(origStage->name(),
                             sio.origInput->desc().dims(), sio.origOutputDesc.dims(),
                             sio.origOutputDesc.dims(),
                             so.kernelSizeX, so.kernelSizeY,
                             so.kernelStride,
                             so.padLeft, so.padRight, so.padTop, so.padBottom, false),
                        direction, tilingsCount) :
                tiler1stAttempt;

        //
        // Use SW stage if tiling optimization failed
        //

        if (!tiler.isTilingPossible()) {
            origStage->attrs().set<bool>("tryHW", false);

            auto swConvOutput = sio.origOutput;
            if (so.withReLU || so.withPool || so.withClamp) {
                swConvOutput = model->addNewData(
                    origStage->name(),
                    sio.origOutputDesc);
                swConvOutput->attrs().copyFrom(sio.origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swConvOutput);
            }

            auto hwPoolInput = swConvOutput;
            if (so.withReLU) {
                auto swReluOutput = sio.origOutput;
                if (so.withPool) {
                    swReluOutput = model->addNewData(
                        origStage->name() + "@ReLU",
                        sio.origOutputDesc);
                    swReluOutput->attrs().copyFrom(sio.origOutput->attrs());
                }

                _stageBuilder->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    so.negativeSlope,
                    swConvOutput,
                    swReluOutput);

                hwPoolInput = swReluOutput;
            }

            if (so.withClamp) {
                auto swClampOutput = sio.origOutput;
                if (so.withPool) {
                    swClampOutput = model->addNewData(
                            origStage->name() + "@Clamp",
                            sio.origOutputDesc);
                    swClampOutput->attrs().copyFrom(sio.origOutput->attrs());
                }

                _stageBuilder->addClampStage(
                        model,
                        origStage->name() + "@Clamp",
                        origStage->origLayer(),
                        0.0,
                        so.clampMax,
                        swConvOutput,
                        swClampOutput);

                hwPoolInput = swClampOutput;
            }

            if (so.withPool) {
                auto hwPoolStage = model->addNewStage<StubStage>(
                    origStage->name() + "@Pool",
                    StageType::StubMaxPool,
                    origStage->origLayer(),
                    {hwPoolInput},
                    {sio.origOutput});

                hwPoolStage->attrs().set<int>("kernelSizeX", so.poolKernelSizeX);
                hwPoolStage->attrs().set<int>("kernelSizeY", so.poolKernelSizeY);

                hwPoolStage->attrs().set<int>("kernelStrideX", so.poolKernelStride);
                hwPoolStage->attrs().set<int>("kernelStrideY", so.poolKernelStride);

                hwPoolStage->attrs().set<int>("padLeft", so.poolPadLeft);
                hwPoolStage->attrs().set<int>("padRight", so.poolPadRight);
                hwPoolStage->attrs().set<int>("padTop", so.poolPadTop);
                hwPoolStage->attrs().set<int>("padBottom", so.poolPadBottom);

                hwPoolStage->attrs().set<bool>("excludePad", false);

                hwPoolStage->attrs().set<bool>("tryHW", true);
            }

            continue;
        }

        model->disconnectStage(origStage);

        for (const auto &tiling : tiler.getHwTilings()) {
            HWConvStageTiler hwStageTiler(so, sio, model,
                    origStage, _stageBuilder, tiling, so.withPool && !tiler.withPool());

            //
            // Split/concat input/output tiles
            //

            if (!hwStageTiler.hwInputTiles.empty()) {
                _stageBuilder->addSplitStage(
                    model,
                    origStage->name() + "@split-input",
                    origStage->origLayer(),
                    std::move(hwStageTiler.hwInputTilesOffsets),
                    hwStageTiler.hwInput,
                    hwStageTiler.hwInputTiles);
            }

            if (!hwStageTiler.hwOutputTiles.empty()) {
                _stageBuilder->addConcatStage(
                    model,
                    origStage->name() + "@concat-output",
                    origStage->origLayer(),
                    std::move(hwStageTiler.hwOutputTilesOffsets),
                    hwStageTiler.hwOutputTiles,
                    hwStageTiler.hwOutput);
            }
        }

        //
        // Remove original stage
        //

        model->removeStage(origStage);
    }
}

}  // namespace

Pass::Ptr PassManager::hwConvTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
