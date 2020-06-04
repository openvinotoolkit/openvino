// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"

#include <set>
#include <memory>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override {
        for (const auto& data : model->datas()) {
            if (data->usage() != DataUsage::Output || data->parentDataToShapeEdge() != nullptr) {
                continue;
            }

            const auto& producer = data->producer();
            VPU_THROW_UNLESS(producer, "Output data must have a producer, but {} doesn't have", data->name());

            if (producer->type() != StageType::Convert) {
                continue;
            }

            VPU_THROW_UNLESS(producer->numInputs() == 1,
                "Only single input producers are supported, but {} has {} inputs",
                producer->name(), producer->numInputs());

            const auto& input = producer->input(0);
            const auto& parentDataToShapeEdge = input->parentDataToShapeEdge();
            if (parentDataToShapeEdge == nullptr) {
                continue;
            }
            const auto parent = parentDataToShapeEdge->parent();

            // MyriadInferRequest::GetResult assumes that dynamic data object has shape data object
            // with the same name + suffix "@shape"
            const auto shapeName = data->name() + "@shape";
            const auto& shapeOutput = model->addOutputData(shapeName, parent->desc());

            if (parent->numConsumers() > 0) {
                _stageBuilder->addCopyStage(
                    model,
                    "copy-for-dynamic-output",
                    nullptr,
                    parent,
                    shapeOutput,
                    "PropagateDynamismToOutput");

            } else {
                const auto parentProducerEdge = parent->producerEdge();
                VPU_THROW_UNLESS(parentProducerEdge != nullptr,
                    "Data containing shape is expected to have a producer, but {} doesn't have", parent->name());

                for (const auto& dataToShapeEdge : parent->childDataToShapeEdges()) {
                    model->replaceDataToShapeParent(dataToShapeEdge, shapeOutput);
                }

                model->replaceStageOutput(parentProducerEdge, shapeOutput);
                model->removeUnusedData(parent);
            }

            model->connectDataWithShape(shapeOutput, data);
        }
    }

private:
    StageBuilder::Ptr _stageBuilder;
};

}  // namespace

Pass::Ptr PassManager::propagateDynamismToOutputs() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
