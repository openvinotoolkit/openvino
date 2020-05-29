// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

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
    VPU_PROFILE(convertShapeNotation);

    // Save all the datas to process before the main loop as in the main loop we will
    // modify the datas list and we will get into the infinite loop
    DataVector shapes;

    for (const auto& data : model->datas()) {
        if (!data->childDataToShapeEdges().empty()) {
            shapes.push_back(data);
        }
    }

    for (const auto& shape : shapes) {
        // Revert shape from IE to MDK notation
        auto convertedShape = model->duplicateData(shape, "@converted-notation");

        const auto generator = [&convertedShape](const ie::Blob::Ptr& blob) {
            std::vector<int32_t> gatherIndices(static_cast<size_t>(convertedShape->desc().totalDimSize()));
            std::iota(gatherIndices.rbegin(), gatherIndices.rend(), 0);

            auto buffer = blob->buffer().as<int32_t*>();

            std::copy(gatherIndices.begin(), gatherIndices.end(), buffer);
        };

        auto gatherIndices = model->addConstData(shape->name() + "@gather-indices",
                                                 DataDesc(DataType::S32, DimsOrder::C, {convertedShape->desc().totalDimSize()}),
                                                 generator);

        _stageBuilder->addGatherStage(model,
                                      shape->name() + "@convert-notation",
                                      nullptr,
                                      shape,
                                      gatherIndices,
                                      convertedShape,
                                      Dim::C);

        for (const auto& dataToShapeEdge : shape->childDataToShapeEdges()) {
            model->replaceDataToShapeParent(dataToShapeEdge, convertedShape);
        }

        // In case if data and shape had the same producer
        // Topological order (nextStages/previousStages) needs to be updated
        for (const auto& dataToShapeEdge : convertedShape->childDataToShapeEdges()) {
            const auto& child = dataToShapeEdge->child();

            if (!child->producer() || child->producer() != shape->producer()) {
                continue;
            }

            const auto& dependentStagesEdges = convertedShape->dependentStagesEdges();

            for (const auto& consumer : child->consumers()) {
                const auto it = std::find_if(dependentStagesEdges.begin(), dependentStagesEdges.end(), [&consumer](const StageDependency& edge) {
                    return edge->dependentStage() == consumer;
                });

                if (it != dependentStagesEdges.end()) {
                    continue;
                }

                model->addStageDependency(consumer, convertedShape);
            }
        }
    }
}
}  // namespace

Pass::Ptr PassManager::convertShapeNotation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
