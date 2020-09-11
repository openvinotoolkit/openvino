// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>

namespace vpu {
namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override {
        VPU_PROFILE(addCopyForOutputsInsideNetwork);

        for (const auto& outputData : model->datas()) {
            if (outputData->usage() != DataUsage::Output || outputData->numConsumers() == 0) {
                continue;
            }

            VPU_THROW_UNLESS(outputData->childDataToShapeEdges().empty(),
                "Output data object cannot be a shape parent for another data object, since notation conversion "
                " has already happened, but {} with usage {} has {} data to shape children",
                outputData->name(), outputData->usage(), outputData->childDataToShapeEdges().size());

            auto newIntermediateData = model->duplicateData(
                outputData,
                "@intermediate",
                outputData->desc());

            auto producer = outputData->producerEdge();
            model->replaceStageOutput(producer, newIntermediateData);
            for (auto consumerEdge : outputData->consumerEdges()) {
                model->replaceStageInput(consumerEdge, newIntermediateData);
            }

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@copy-to-output", outputData->name()),
                nullptr,
                newIntermediateData,
                outputData,
                "addCopyForOutputsInsideNetwork");

            if (const auto& parentShapeEdge = outputData->parentDataToShapeEdge()) {
                model->connectDataWithShape(parentShapeEdge->parent(), newIntermediateData);
            }
        }
    }

private:
    StageBuilder::Ptr _stageBuilder;
};

}  // namespace

Pass::Ptr PassManager::addCopyForOutputsInsideNetwork() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
