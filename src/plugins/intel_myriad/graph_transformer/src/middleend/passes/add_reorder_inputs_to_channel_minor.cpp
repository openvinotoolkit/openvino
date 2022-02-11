// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>
#include <utility>

namespace vpu {
namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override {
        VPU_PROFILE(addChannelMinorReorderForInputs);

        for (auto inputData : model->datas()) {
            if (inputData->usage() != DataUsage::Input) {
                continue;
            }

            // Floating point Data are converted to fp16, add reorder to fp16 part in this case
            inputData = inputData->attrs().getOrDefault("fp16_copy", inputData);

            auto dataDesc = inputData->desc();
            if (dataDesc.numDims() < 3 || dataDesc.numDims() > 4) continue;
            if (dataDesc.dimsOrder().dimInd(Dim::C) == 0) continue;
            dataDesc.moveDim(Dim::C, 0);

            const auto newIntermediateData = model->duplicateData(
                    inputData,
                    "@intermediate",
                    dataDesc);

            for (const auto& consumerEdge : inputData->consumerEdges()) {
                model->replaceStageInput(consumerEdge, newIntermediateData);
            }

            _stageBuilder->addReorderStage(
                    model,
                    formatString("%s@reorder=%s", inputData->name(), dataDesc.dimsOrder()),
                    nullptr,
                    inputData,
                    newIntermediateData);
        }
    }

private:
    StageBuilder::Ptr _stageBuilder;
};

}  // namespace

Pass::Ptr PassManager::reorderInputsToChannelMinor() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
