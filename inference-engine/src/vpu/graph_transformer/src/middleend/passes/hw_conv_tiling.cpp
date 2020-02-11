// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <precision_utils.h>
#include <utility>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/middleend/hw/conv_tiling/hw_convolution_tiler.hpp>
#include <vpu/middleend/hw/conv_tiling/hw_stage_tiler.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(hwConvTiling);

    for (const auto& origStage : model->getStages()) {
        if (origStage->type() != StageType::StubConv) {
            continue;
        }

        const auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        const HWConvStageOptions stageOptions(origStage);
        const HWConvStageIO stageIO(origStage, origStage->output(0));

        //
        // Unsupported paddings
        //

        //
        // Try to find "best" tiling
        //

        const size_t tilingsCount = 1;
        const HWTilingNS::Direction direction = HWTilingNS::Direction::INPUT_TO_OUTPUT;
                                             // HWTilingNS::Direction::OUTPUT_TO_INPUT;

        const auto convolutionOptions = HWTilingNS::ConvolutionOptions{
            origStage->name(),
            stageIO.origInput->desc().dims(),
            stageIO.origOutput->desc().dims(),
            stageIO.origOutputDesc.dims(),
            stageOptions.kernelSizeX,
            stageOptions.kernelSizeY,
            stageOptions.kernelStride,
            stageOptions.padLeft,
            stageOptions.padRight,
            stageOptions.padTop,
            stageOptions.padBottom,
            stageOptions.withPool
        };

        const HWTilingNS::HWConvolutionTiler tiler1stAttempt(convolutionOptions, direction, tilingsCount);


        const HWTilingNS::HWConvolutionTiler& tiler = [&] {
            if (!tiler1stAttempt.isTilingPossible() && tiler1stAttempt.withPool()) {
                const auto optionsWithoutPool = HWTilingNS::ConvolutionOptions{
                    origStage->name(),
                    stageIO.origInput->desc().dims(),
                    stageIO.origOutputDesc.dims(),
                    stageIO.origOutputDesc.dims(),
                    stageOptions.kernelSizeX,
                    stageOptions.kernelSizeY,
                    stageOptions.kernelStride,
                    stageOptions.padLeft,
                    stageOptions.padRight,
                    stageOptions.padTop,
                    stageOptions.padBottom,
                    false
                };

                return HWTilingNS::HWConvolutionTiler{optionsWithoutPool, direction, tilingsCount};
            } else {
                return tiler1stAttempt;
            }
        }();

        //
        // Use SW stage if tiling optimization failed
        //

        if (!tiler.isTilingPossible()) {
            origStage->attrs().set<bool>("tryHW", false);

            auto swConvOutput = stageIO.origOutput;
            if (stageOptions.withReLU || stageOptions.withPool || stageOptions.withClamp) {
                swConvOutput = model->addNewData(origStage->name(), stageIO.origOutputDesc);
                swConvOutput->attrs().copyFrom(stageIO.origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swConvOutput);
            }

            auto hwPoolInput = swConvOutput;
            if (stageOptions.withReLU) {
                auto swReluOutput = stageIO.origOutput;
                if (stageOptions.withPool) {
                    swReluOutput = model->addNewData(origStage->name() + "@ReLU", stageIO.origOutputDesc);
                    swReluOutput->attrs().copyFrom(stageIO.origOutput->attrs());
                }

                _stageBuilder->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    stageOptions.negativeSlope,
                    swConvOutput,
                    swReluOutput);

                hwPoolInput = swReluOutput;
            }

            if (stageOptions.withClamp) {
                auto swClampOutput = stageIO.origOutput;
                if (stageOptions.withPool) {
                    swClampOutput = model->addNewData(origStage->name() + "@Clamp", stageIO.origOutputDesc);
                    swClampOutput->attrs().copyFrom(stageIO.origOutput->attrs());
                }

                _stageBuilder->addClampStage(
                    model,
                    origStage->name() + "@Clamp",
                    origStage->origLayer(),
                    0.0,
                    stageOptions.clampMax,
                    swConvOutput,
                    swClampOutput);

                hwPoolInput = swClampOutput;
            }

            if (stageOptions.withPool) {
                auto hwPoolStage = model->addNewStage<StubStage>(
                    origStage->name() + "@Pool",
                    StageType::StubMaxPool,
                    origStage->origLayer(),
                    {hwPoolInput},
                    {stageIO.origOutput});

                hwPoolStage->attrs().set<int>("kernelSizeX", stageOptions.poolKernelSizeX);
                hwPoolStage->attrs().set<int>("kernelSizeY", stageOptions.poolKernelSizeY);

                hwPoolStage->attrs().set<int>("kernelStrideX", stageOptions.poolKernelStride);
                hwPoolStage->attrs().set<int>("kernelStrideY", stageOptions.poolKernelStride);

                hwPoolStage->attrs().set<int>("padLeft", stageOptions.poolPadLeft);
                hwPoolStage->attrs().set<int>("padRight", stageOptions.poolPadRight);
                hwPoolStage->attrs().set<int>("padTop", stageOptions.poolPadTop);
                hwPoolStage->attrs().set<int>("padBottom", stageOptions.poolPadBottom);

                hwPoolStage->attrs().set<bool>("excludePad", false);

                hwPoolStage->attrs().set<bool>("tryHW", true);
            }

            continue;
        }

        model->disconnectStage(origStage);

        for (const auto &tiling : tiler.getHwTilings()) {
            HWConvStageTiler hwStageTiler(
                stageOptions,
                stageIO,
                model,
                origStage,
                _stageBuilder,
                tiling,
                stageOptions.withPool && !tiler.withPool());

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
