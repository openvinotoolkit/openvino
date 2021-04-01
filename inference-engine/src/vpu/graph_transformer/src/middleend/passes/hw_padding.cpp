// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <cmath>

#include <tuple>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuidler) : _stageBuilder(stageBuidler) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

bool supportedPaddingPool(const Stage& stage) {
    IE_ASSERT(StageType::StubMaxPool == stage->type() ||
              StageType::StubAvgPool == stage->type());

    auto input  = stage->input(0);
    auto output = stage->output(0);

    auto kernelSizeX  = stage->attrs().get<int>("kernelSizeX");
    auto kernelSizeY  = stage->attrs().get<int>("kernelSizeY");
    auto kernelStride = stage->attrs().get<int>("kernelStrideX");
    auto padLeft      = stage->attrs().get<int>("padLeft");
    auto padRight     = stage->attrs().get<int>("padRight");
    auto padTop       = stage->attrs().get<int>("padTop");
    auto padBottom    = stage->attrs().get<int>("padBottom");

    //
    // Even kernel size with odd input -> HW bug
    // Need to add extra border
    //

    bool forcePaddingStage = false;

    if (kernelSizeX % 2 == 0 && input->desc().dim(Dim::W) % 2 == 1) {
        if (padRight == 0) {
            stage->attrs().set<int>("padRight", 1);
        }

        forcePaddingStage = true;
    }

    if (kernelSizeY % 2 == 0 && input->desc().dim(Dim::H) % 2 == 1) {
        if (padBottom == 0) {
            stage->attrs().set<int>("padBottom", 1);
        }

        forcePaddingStage = true;
    }

    auto hwInitialPad = getHwPaddingInfo(
        input->desc().dims(), output->desc().dims(),
        kernelSizeX, kernelSizeY,
        kernelStride, kernelStride,
        padLeft, padTop);

    //
    // HW unit supports pooling with even-sized kernel with such asymmetrical paddings.
    // But it does not support inverted paddings.
    // For odd-sized kernels supported paddings are symmetrical.
    //

    bool isPadSupported =
        (hwInitialPad.left   == 0 || hwInitialPad.left   == kernelSizeX / 2)       &&
        (hwInitialPad.right  == 0 || hwInitialPad.right  == (kernelSizeX - 1) / 2) &&
        (hwInitialPad.top    == 0 || hwInitialPad.top    == kernelSizeY / 2)       &&
        (hwInitialPad.bottom == 0 || hwInitialPad.bottom == (kernelSizeY - 1) / 2);

    return isPadSupported && !forcePaddingStage;
}

bool supportedPaddingConv(const Stage& stage) {
    IE_ASSERT(StageType::StubConv == stage->type());

    auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
    auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
    auto padLeft     = stage->attrs().get<int>("padLeft");
    auto padRight    = stage->attrs().get<int>("padRight");
    auto padTop      = stage->attrs().get<int>("padTop");
    auto padBottom   = stage->attrs().get<int>("padBottom");

    //
    // HW unit supports convolution with even-sized kernel with such asymmetrical paddings.
    // But it does not support inverted paddings.
    // For odd-sized kernels supported paddings are symmetrical.
    //

    bool paddingsAreZeros = padLeft == 0 && padTop == 0 && padRight == 0 && padBottom == 0;
    bool paddingsAreSupported =
        padLeft   == kernelSizeX / 2 &&
        padTop    == kernelSizeY / 2 &&
        padRight  == (kernelSizeX - 1) / 2 &&
        padBottom == (kernelSizeY - 1) / 2;

    return paddingsAreZeros || paddingsAreSupported;
}

void insertPaddingStageBefore(const Model& model, StageBuilder::Ptr& stageBuilder, const Stage& origStage) {
    auto origInput       = origStage->input(0);
    auto paddedInputDesc = origInput->desc();

    auto padLeft   = origStage->attrs().get<int>("padLeft");
    auto padRight  = origStage->attrs().get<int>("padRight");
    auto padTop    = origStage->attrs().get<int>("padTop");
    auto padBottom = origStage->attrs().get<int>("padBottom");

    paddedInputDesc.setDim(Dim::W, origInput->desc().dim(Dim::W) + padLeft + padRight);
    paddedInputDesc.setDim(Dim::H, origInput->desc().dim(Dim::H) + padTop + padBottom);

    auto inputPadded = model->duplicateData(
        origInput,
        "@padded",
        paddedInputDesc);

    model->replaceStageInput(origStage->inputEdge(0), inputPadded);

    auto paddingStage = stageBuilder->addPadStage(
        model,
        origStage->name() + "@padding",
        origStage->origLayer(),
        (origStage->type() == StageType::StubMaxPool) ? PadMode::Edge : PadMode::Constant,
        0.0f,
        DimValues({
            {Dim::W, padLeft},
            {Dim::H, padTop},
        }),
        DimValues({
            {Dim::W, padRight},
            {Dim::H, padBottom},
        }),
        origInput,
        inputPadded);

    origStage->attrs().set<int>("padLeft",   0);
    origStage->attrs().set<int>("padRight",  0);
    origStage->attrs().set<int>("padTop",    0);
    origStage->attrs().set<int>("padBottom", 0);
}

void PassImpl::run(const Model& model) {
    VPU_PROFILE(hwPadding);

    auto isPooling = [](const Stage& stage) {
        return StageType::StubMaxPool == stage->type() ||
               StageType::StubAvgPool == stage->type();
    };
    auto isConv = [](const Stage& stage) {
        return StageType::StubConv == stage->type();
    };

    auto stages = model->getStages();

    for (const auto& origStage : stages) {
        if (!isPooling(origStage) && !isConv(origStage)) {
            continue;
        }

        auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        bool addPaddingStage = false;

        if (isConv(origStage)) {
            addPaddingStage = !supportedPaddingConv(origStage);
        } else if (isPooling(origStage)) {
            addPaddingStage = !supportedPaddingPool(origStage);
        } else {
            IE_ASSERT(false);
        }

        if (addPaddingStage) {
            insertPaddingStageBefore(model, _stageBuilder, origStage);
        }
    }
}

}  // namespace

Pass::Ptr PassManager::hwPadding() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
