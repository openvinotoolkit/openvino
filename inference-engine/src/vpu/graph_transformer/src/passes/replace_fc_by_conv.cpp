// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stub_stage.hpp>

namespace vpu {

namespace {

void setConvParameters(const vpu::Stage& stage, int kX, int kY) {
    stage->attrs().set("kernelSizeX", kX);
    stage->attrs().set("kernelSizeY", kY);

    stage->attrs().set("kernelStrideX", kX);
    stage->attrs().set("kernelStrideY", kY);

    stage->attrs().set("padLeft", 0);
    stage->attrs().set("padRight", 0);
    stage->attrs().set("padTop", 0);
    stage->attrs().set("padBottom", 0);

    stage->attrs().set("dilationX", 1);
    stage->attrs().set("dilationY", 1);

    stage->attrs().set("groupSize", 1);

    stage->attrs().set("tryHW", true);
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(replaceFCbyConv);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubFullyConnected) {
            continue;
        }

        const auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        const auto input = stage->input(0);
        const auto weights = stage->input(1);
        const auto biases  = stage->input(2);
        const auto output = stage->output(0);

        const auto inDims = input->desc().dims();

        if (inDims.size() != 2 && inDims.size() != 4) {
            continue;
        }

        const auto inBatch = inDims[Dim::N];
        const auto inSize  = input->desc().totalDimSize() / inBatch;

        IE_ASSERT(output->desc().dim(Dim::N) == inBatch);

        // HW restriction for kernel stride (we use stride equal to kernel size).
        const int maxKernelSize = 8;

        // TODO: something more sophisticated?
        int convKernelSizeX = -1;
        int convKernelSizeY = -1;
        for (int k = maxKernelSize; k >= 1; --k) {
            if (inSize >= (k * k) && inSize % (k * k) == 0 && isPowerOfTwo(inSize / (k * k))) {
                convKernelSizeX = k;
                convKernelSizeY = k;
                break;
            }
        }
        if (convKernelSizeX == -1 || convKernelSizeY == -1) {
            for (int k = maxKernelSize; k >= 1; --k) {
                if (inSize >= (k * k) && inSize % (k * k) == 0) {
                    convKernelSizeX = k;
                    convKernelSizeY = k;
                    break;
                }
            }
        }

        if (convKernelSizeX == -1 || convKernelSizeY == -1) {
            continue;
        }

        const auto convInputC = inSize / (convKernelSizeX * convKernelSizeY);

        model->disconnectStage(stage);

        // TODO: something more sophisticated?
        int batchStepW = 1;
        int batchStepH = 1;
        for (auto div : {100, 50, 20, 10}) {
            if (inBatch >= div && inBatch % div == 0) {
                batchStepW = div;
                batchStepH = inBatch / div;
                break;
            }
        }

        Data convInput;
        if (batchStepW == 1 && batchStepH == 1) {
            convInput = model->duplicateData(
                input,
                "@reshape",
                DataDesc{convKernelSizeX, convKernelSizeY, convInputC, inBatch});

            _stageBuilder->addReshapeStage(
                model,
                convInput->name(),
                stage->origLayer(),
                input,
                convInput);
        } else {
            // NCDHW
            const auto reshaped = model->duplicateData(
                input,
                "@reshape",
                DataDesc{convKernelSizeX, convKernelSizeY, convInputC, batchStepW, batchStepH});

            _stageBuilder->addReshapeStage(
                model,
                reshaped->name(),
                stage->origLayer(),
                input,
                reshaped);

            // NCDHW
            const auto permuted = model->duplicateData(
                input,
                "@permute-batch",
                DataDesc{convKernelSizeX, batchStepW, convKernelSizeY, batchStepH, convInputC});

            _stageBuilder->addPermuteStage(
                model,
                permuted->name(),
                stage->origLayer(),
                reshaped,
                permuted,
                DimValues_<Dim>{{Dim::W, Dim::W}, {Dim::H, Dim::C}, {Dim::D, Dim::H}, {Dim::C, Dim::N}, {Dim::N, Dim::D}});

            // NCHW
            const auto merged = model->duplicateData(
                input,
                "@merge-batch",
                DataDesc{convKernelSizeX * batchStepW, convKernelSizeY * batchStepH, convInputC, 1});

            _stageBuilder->addReshapeStage(
                model,
                merged->name(),
                stage->origLayer(),
                permuted,
                merged);

            convInput = merged;
        }

        Data convOutput;
        if (batchStepW == 1 && batchStepH == 1) {
            convOutput = model->duplicateData(
                output,
                "@reshape",
                DataDesc{1, 1, output->desc().dim(Dim::C), inBatch});

            _stageBuilder->addReshapeStage(
                model,
                convOutput->name(),
                stage->origLayer(),
                convOutput,
                output);
        } else {
            // NCDHW
            const auto reshaped = model->duplicateData(
                output,
                "@reshape",
                DataDesc{1, 1, output->desc().dim(Dim::C), batchStepW, batchStepH});

            _stageBuilder->addReshapeStage(
                model,
                reshaped->name(),
                stage->origLayer(),
                reshaped,
                output);

            // NCDHW
            const auto permuted = model->duplicateData(
                output,
                "@permute-batch",
                DataDesc{1, batchStepW, 1, batchStepH, output->desc().dim(Dim::C)});

            _stageBuilder->addPermuteStage(
                model,
                permuted->name(),
                stage->origLayer(),
                permuted,
                reshaped,
                DimValues_<Dim>{{Dim::W, Dim::W}, {Dim::H, Dim::D}, {Dim::D, Dim::N}, {Dim::C, Dim::H}, {Dim::N, Dim::C}});

            // NCHW
            const auto merged = model->duplicateData(
                output,
                "@merge-batch",
                DataDesc{batchStepW, batchStepH, output->desc().dim(Dim::C), 1});

            _stageBuilder->addReshapeStage(
                model,
                merged->name(),
                stage->origLayer(),
                merged,
                permuted);

            convOutput = merged;
        }

        const auto convWeights = model->duplicateData(
            weights,
            "@fc-to-conv",
            DataDesc({
                convKernelSizeX,
                convKernelSizeY,
                convInputC,
                output->desc().dim(Dim::C)}));

        auto convStage = model->addNewStage<StubStage>(
            stage->name() + "@fc-to-conv",
            StageType::StubConv,
            stage->origLayer(),
            {convInput, convWeights, biases},
            {convOutput});
        convStage->attrs().copyFrom(stage->attrs());
        setConvParameters(convStage, convKernelSizeX, convKernelSizeY);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::replaceFCbyConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
