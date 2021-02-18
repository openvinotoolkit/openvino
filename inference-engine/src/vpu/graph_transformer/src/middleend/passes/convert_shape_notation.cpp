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

        // Settings IE-notation attribute to shape must be done after duplicateData
        // Since duplicateData does deep attributes copy
        shape->attrs().set<bool>("IE-notation", true);
        convertedShape->attrs().set<bool>("converted-notation", true);

        const auto generator = [&convertedShape](const ie::Blob::Ptr& blob) {
            std::vector<int32_t> gatherIndices(static_cast<size_t>(convertedShape->desc().totalDimSize()));
            std::iota(gatherIndices.rbegin(), gatherIndices.rend(), 0);

            auto buffer = blob->buffer().as<int32_t*>();

            std::copy(gatherIndices.begin(), gatherIndices.end(), buffer);
        };

        auto gatherIndices = model->addConstData(shape->name() + "@gather-indices",
                                                 DataDesc(DataType::S32, DimsOrder::C, {convertedShape->desc().totalDimSize()}),
                                                 generator);

        const auto& gather = _stageBuilder->addGatherStage(
            model,
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
        // Topological order (nextStages/previousStages) needs to be updated.
        // Also it is needed if data is the network Input data.
        for (const auto& dataToShapeEdge : convertedShape->childDataToShapeEdges()) {
            const auto& child = dataToShapeEdge->child();

            const auto& childProducer = child->producer();
            if (!childProducer) {
                VPU_THROW_UNLESS(child->usage() == DataUsage::Input,
                        "ConvertShapeNotation pass for shape of name {} failed: if child data of name {} "
                        "has no producer than it must have Input data usage, actual: {}",
                        shape->name(), child->name(), child->usage());
            } else if (child->producer() != shape->producer()) {
                continue;
            }

            const auto& stageDependencyEdges = gather->childDependencyEdges();

            for (const auto& consumer : child->consumers()) {
                const auto it = std::find_if(stageDependencyEdges.begin(), stageDependencyEdges.end(), [&consumer](const StageDependency& edge) {
                    return edge->child() == consumer;
                });

                if (it != stageDependencyEdges.end()) {
                    continue;
                }

                model->addStageDependency(gather, consumer);
            }
        }
    }
}
}  // namespace

Pass::Ptr PassManager::convertShapeNotation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
