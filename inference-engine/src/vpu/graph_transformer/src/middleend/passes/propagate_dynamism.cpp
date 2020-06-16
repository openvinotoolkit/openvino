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
        VPU_PROFILE(propagateDynamism);

        for (const auto& stage : model->getStages()) {
            if (stage->type() != StageType::Copy && stage->type() != StageType::Convert &&
                stage->type() != StageType::Power && stage->type() != StageType::Prod) {
                continue;
            }

            const auto& inputs = stage->inputs();
            std::vector<DataToShapeAllocation> parentInputShapeEdges;

            for (const auto& input : inputs) {
                const auto parentDataToShapeEdge = input->parentDataToShapeEdge();
                if (parentDataToShapeEdge) {
                    parentInputShapeEdges.push_back(parentDataToShapeEdge);
                }
            }

            VPU_THROW_UNLESS(stage->numOutputs() == 1,
                             "PropagateDynamism: only single output stages are supported, but {} stage of name {} has {} inputs",
                             stage->type(), stage->name(), stage->numOutputs());

            const auto& output = stage->output(0);
            const auto& parentOutputShapeEdge = output->parentDataToShapeEdge();

            const auto dynamismIsNotNeeded = parentInputShapeEdges.empty() && !parentOutputShapeEdge;
            const auto dynamismIsAlreadyPropagated = !parentInputShapeEdges.empty() && parentOutputShapeEdge;

            if (dynamismIsNotNeeded || dynamismIsAlreadyPropagated) {
                continue;
            }

            const auto validateShapeConversion = [](const Data& shape) {
                const auto& shapeAttrs = shape->attrs();
                VPU_THROW_UNLESS(shapeAttrs.getOrDefault("converted-notation", false),
                                 "All shape parent data object must be already converted to MDK notation");

                const auto& shapeProducer = shape->producer();
                const auto& shapeInIENotation = shapeProducer->input(0);
                VPU_THROW_UNLESS(shapeInIENotation->usage() == DataUsage::Intermediate || shapeInIENotation->usage() == DataUsage::Input,
                                 "Shape parent data object is expected to be an intermediate data object since shape child is not an output");

                const auto& shapeInIENotationAttrs = shapeInIENotation->attrs();
                VPU_THROW_UNLESS(shapeInIENotationAttrs.getOrDefault("IE-notation", false),
                                 "Unexpected data object as shape in IE notation");
            };

            const auto validateShapes = [&stage](const Data& input, const Data& output) {
                // Propagation is only supported for input and output with equal upper-bound shapes.
                // For example, Prod stage with dynamic input and broadcast is not supported.
                for (const auto& dim : input->desc().dims()) {
                    VPU_THROW_UNLESS(dim.second == output->desc().dim(dim.first),
                                     "PropagateDynamism: {} stage of name {} must have input of name {} with upper-bound dimension {} "
                                     "which should be equal to output which is {}, actual: {}",
                                     stage->type(), stage->name(), input->name(), dim.first,
                                     output->desc().dim(dim.first), dim.second);
                }
            };

            if (!parentInputShapeEdges.empty() && !parentOutputShapeEdge) {
                VPU_THROW_UNLESS(parentInputShapeEdges.size() == 1,
                                 "PropagateDynamism for stage {} of name {} failed: propagation dynamism from multiple inputs is not "
                                 "supported, actual number of dynamic inputs: {}", stage->type(), stage->name(), parentInputShapeEdges.size());
                const auto& parentInputShapeEdge = parentInputShapeEdges[0];
                const auto& input = parentInputShapeEdge->child();
                validateShapes(input, output);

                const auto shape = parentInputShapeEdge->parent();
                validateShapeConversion(shape);

                model->connectDataWithShape(shape, output);

                if (output->usage() == DataUsage::Output) {
                    // MyriadInferRequest::GetResult assumes that dynamic data object has shape data object
                    // with the same name + suffix "@shape"
                    const auto shapeName = output->name() + "@shape";
                    const auto& shapeOutput = model->addOutputData(shapeName, shape->desc());

                    const auto& shapeProducer = shape->producer();
                    const auto& shapeInIENotation = shapeProducer->input(0);

                    _stageBuilder->addCopyStage(
                            model,
                            "copy-for-dynamic-output",
                            nullptr,
                            shapeInIENotation,
                            shapeOutput,
                            "PropagateDynamismToOutput");
                }
            } else if (parentInputShapeEdges.empty() && parentOutputShapeEdge) {
                VPU_THROW_UNLESS(inputs.size() == 1,
                                 "PropagateDynamism for stage {} of name {} failed: propagation dynamism from output "
                                 "to multiple inputs is not supported", stage->type(), stage->name());
                const auto& input = inputs[0];
                validateShapes(input, output);

                const auto shape = parentOutputShapeEdge->parent();
                validateShapeConversion(shape);

                VPU_THROW_UNLESS(input->usage() == DataUsage::Input,
                    "PropagateDynamism for stage {} of type {} failed: propagate output dynamism to "
                    "input with name {} is available for data with only Input data usage, actual: {}",
                    stage->name(), stage->type(), input->name(), shape->usage());

                model->connectDataWithShape(shape, input);
            }
        }
    }

private:
    StageBuilder::Ptr _stageBuilder;
};

}  // namespace

Pass::Ptr PassManager::propagateDynamism() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
