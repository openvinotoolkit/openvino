// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stages/stub_stage.hpp>
#include <vpu/middleend/sw/utility.hpp>
#include <vpu/compile_env.hpp>


namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(replaceGemmByConv);

    auto stages = model->getStages();
    for (const auto& stage : stages) {
        if (stage->type() != StageType::GEMM) {
            continue;
        }

        const auto transposeA = stage->attrs().get<bool>("transposeA");
        const auto transposeB = stage->attrs().get<bool>("transposeB");
        const auto alpha = stage->attrs().get<float>("alpha");
        const auto beta = stage->attrs().get<float>("beta");

        const auto inputA = stage->input(0);
        auto inputB = stage->input(1);
        const auto output = stage->output(0);

        const auto inputDimsA = inputA->desc().dims();
        const auto inputDimsB = inputB->desc().dims();
        const auto outputDims = output->desc().dims();

        const auto K = inputDimsA[Dim::W];
        const auto M = inputDimsA[Dim::H];
        const auto N = transposeB ? inputDimsB[Dim::H] : inputDimsB[Dim::W];
        const auto batches = inputDimsA[Dim::C];

        // batch by "N" dimension is not supported
        if (!(inputDimsA[Dim::N] == 1 && inputDimsB[Dim::N] == 1 && outputDims[Dim::N] == 1)) continue;
        // "3 inputs" is not supported
        if (stage->inputs().size() == 3) continue;
        // cases with tiling and unaligned "M" are not supported
        if (!(M % 8 == 0 && K % 8 == 0 && alpha == 1.0f && beta == 1.0f)) continue;
        /* "transposeA" case is not supported */
        if (transposeA) continue;
        if (!(batches == inputDimsB[Dim::C] && batches == outputDims[Dim::C])) continue;

        const auto gemmScale = stage->attrs().getOrDefault<float>("scaleFactor", 1.0f);

        DataVector subInputsA(batches);
        DataVector subInputsB(batches);
        DataVector subOutputs(batches);

        model->disconnectStage(stage);

        if (transposeB) {
            const auto inputBTransposed = model->duplicateData(inputB, "@transposeB", DataDesc{N, K, batches, 1});
            _stageBuilder->addPermuteStage(
                model,
                stage->name() + "@transposeB",
                stage->origLayer(),
                inputB,
                inputBTransposed,
                DimValues_<Dim>{{Dim::N, Dim::N}, {Dim::H, Dim::W}, {Dim::W, Dim::H},   {Dim::D, Dim::D}, {Dim::C, Dim::C}});
            inputB = inputBTransposed;
        }

        /* C = A * B;
           C = Transpose(Transpose(B) * Transpose(A));
           C = A * B = Transpose(HWConv(Transpose(A), WeightsRepack(B))))

           C = Transpose(Transpose(HWConv(Transpose(Transpose(B)), WeightsRepack(Transpose(A))))) =
             = HWConv(B, WeightsRepack(Transpose(A)))

           Permutation of matrix "A" is added during the tiling stage
        */

        for (int batchIndex = 0; batchIndex < batches; batchIndex++) {
            const auto postfix = formatString("@batchByChannel=%d/%d", batchIndex + 1, batches);

            auto subInputADesc = inputA->desc();
            auto subInputBDesc = inputB->desc();
            subInputADesc.setDim(Dim::C, 1);
            subInputBDesc.setDim(Dim::C, 1);

            auto subOutputDesc = output->desc();
            subOutputDesc.setDim(Dim::C, 1);

            subInputsA[batchIndex] = model->duplicateData(inputA, postfix, subInputADesc);
            subInputsB[batchIndex] = model->duplicateData(inputB, postfix, subInputBDesc);
            subOutputs[batchIndex] = model->duplicateData(output, postfix, subOutputDesc);

            const auto convolvedB = model->duplicateData(subInputsB[batchIndex], "@reshape", DataDesc{N, 1, K, 1});

            const auto convolvedOutput = model->duplicateData(output, "@reshape", DataDesc{N, 1, M, 1});
            const auto convolvedOutputCopy = model->duplicateData(output, "@reshape", DataDesc{N, M, 1, 1});

            _stageBuilder->addReshapeStage(
                model,
                stage->name() + postfix + "@reshapeB",
                stage->origLayer(),
                subInputsB[batchIndex],
                convolvedB);

            auto convStage = model->addNewStage<StubStage>(
                stage->origLayerName() + postfix + "@GEMM_HWConv",
                StageType::StubConv,
                stage->origLayer(),
                {convolvedB, subInputsA[batchIndex], model->addFakeData(), model->addFakeData()},
                {convolvedOutput});

            _stageBuilder->addReshapeStage(
                model,
                stage->name() + postfix + "@reshapeOutput",
                stage->origLayer(),
                convolvedOutput,
                subOutputs[batchIndex]);

            convStage->attrs().set<int>("kernelSizeX", 1);
            convStage->attrs().set<int>("kernelSizeY", 1);
            convStage->attrs().set<int>("kernelStrideX", 1);
            convStage->attrs().set<int>("kernelStrideY", 1);
            convStage->attrs().set<int>("padLeft", 0);
            convStage->attrs().set<int>("padRight", 0);
            convStage->attrs().set<int>("padTop", 0);
            convStage->attrs().set<int>("padBottom", 0);
            convStage->attrs().set<int>("dilationX", 1);
            convStage->attrs().set<int>("dilationY", 1);
            convStage->attrs().set<int>("groupSize", 1);
            convStage->attrs().set<bool>("tryHW", true);
            convStage->attrs().set<float>("scaleFactor", gemmScale);
        }

        _stageBuilder->addSplitStage(
            model,
            stage->name() + "@splitA",
            stage->origLayer(),
            Dim::C,
            inputA,
            subInputsA);

        _stageBuilder->addSplitStage(
            model,
            stage->name() + "@splitB",
            stage->origLayer(),
            Dim::C,
            inputB,
            subInputsB);

        _stageBuilder->addConcatStage(
            model,
            stage->name() + "@concat",
            stage->origLayer(),
            Dim::C,
            subOutputs,
            output);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::replaceGemmByConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
