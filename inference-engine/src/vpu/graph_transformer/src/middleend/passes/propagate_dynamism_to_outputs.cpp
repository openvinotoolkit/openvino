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
        static const std::unordered_set<StageType> types = {StageType::Convert};

        for (auto const& data : model->datas()) {
            if (data->usage() != DataUsage::Output || data->parentDataToShapeEdge() != nullptr) {
                continue;
            }

            auto const& producer = data->producer();
            VPU_THROW_UNLESS(producer, "Output data must have a producer, but {} doesn't have", data->name());

            if (types.count(producer->type()) == 0) {
                continue;
            }

            VPU_THROW_UNLESS(producer->numInputs() == 1,
                "Only single input producers are supported, but {} has {} inputs",
                producer->name(), producer->numInputs());

            auto const& input = producer->input(0);
            auto const& parentDataToShapeEdge = input->parentDataToShapeEdge();
            if (parentDataToShapeEdge == nullptr) {
                continue;
            }
            auto const parent = parentDataToShapeEdge->parent();

            // Create the second output with shape in case of dynamic output
            const auto& shapeOutput = model->addOutputData(parent->name() + "@output-dynamic-shape", parent->desc());

            if (parent->numConsumers() > 0) {
                _stageBuilder->addCopyStage(
                    model,
                    "copy-for-dynamic-output",
                    nullptr,
                    parent,
                    shapeOutput,
                    "PropogateDynamismToOutput");

            } else {
                auto const parentProducerEdge = parent->producerEdge();
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
