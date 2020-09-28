// Copyright (C) 2018-2020 Intel Corporation
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
    VPU_PROFILE(gemmTranspose);

    auto stages = model->getStages();
    for (const auto& stage : stages) {
        if (stage->type() != StageType::GEMM) {
            continue;
        }

        auto transposeA = stage->attrs().get<bool>("transposeA");
        if (!transposeA) continue;

        auto inputA = stage->input(0);
        auto inputB = stage->input(1);
        auto output = stage->output(0);

        VPU_THROW_UNLESS(inputA->parentDataToShapeEdge() == nullptr,
            "Processing layer {} with type {} failed: first input ({} with usage {}) which is dynamic "
            "doesn't support transpose parameter",
            stage->name(), stage->type(), inputA->name(), inputA->usage());

        const auto inputDimsA = inputA->desc().dims();

        const auto K = inputDimsA[Dim::H];
        const auto M = inputDimsA[Dim::W];
        const auto batch1 = inputDimsA[Dim::N];
        const auto batch2 = inputDimsA[Dim::C];

        const auto inputATranspose = model->duplicateData(inputA, "@reshape", DataDesc{K, M, batch2, batch1});

        stage->attrs().set<bool>("transposeA", false);
        model->replaceStageInput(stage->inputEdge(0), inputATranspose);

        _stageBuilder->addPermuteStage(
            model,
            stage->name() + "@transpose",
            stage->origLayer(),
            inputA,
            inputATranspose,
            DimValues_<Dim>{{Dim::W, Dim::H}, {Dim::H, Dim::W}, {Dim::D, Dim::D}, {Dim::C, Dim::C}, {Dim::N, Dim::N}});
    }
}

}  // namespace

Pass::Ptr PassManager::gemmTranspose() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu