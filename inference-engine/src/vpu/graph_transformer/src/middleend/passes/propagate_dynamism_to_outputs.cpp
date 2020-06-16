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

            const auto& parentAttrs = parent->attrs();
            VPU_THROW_UNLESS(parentAttrs.getOrDefault("converted-notation", false),
                "All shape parent data object must be already converted to MDK notation, but {} is in IE notation",
                parent->name());

            const auto& parentInIENotation = parent->producer()->input(0);
            const auto& parentInIENotationAttrs = parentInIENotation->attrs();
            VPU_THROW_UNLESS(parentInIENotationAttrs.getOrDefault("IE-notation", false),
                 "Data object {} is expected to be shape in IE notation, but is not marked as it",
                 parentInIENotation->name());

            VPU_THROW_UNLESS(parentInIENotation->usage() == DataUsage::Intermediate,
                "Shape data object in IE notation {} is expected to be an {} data object, but it has usage {}",
                parentInIENotation->name(), DataUsage::Intermediate, parentInIENotation->usage());

            model->connectDataWithShape(parent, data);

            // MyriadInferRequest::GetResult assumes that dynamic data object has shape data object
            // with the same name + suffix "@shape"
            const auto shapeName = data->name() + "@shape";
            const auto& shapeOutput = model->addOutputData(shapeName, parentInIENotation->desc());

            _stageBuilder->addCopyStage(
                model,
                "copy-for-dynamic-output",
                nullptr,
                parentInIENotation,
                shapeOutput,
                "PropagateDynamismToOutput");
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
