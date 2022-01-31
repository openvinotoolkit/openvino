// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"

#include "vpu/utils/shape_io.hpp"

#include <set>
#include <memory>

namespace vpu {

namespace {

static const std::set<StageType> stagesSupportedInToOutPropagation = {
        StageType::Convert,
        StageType::Copy,
        StageType::Power,
        StageType::Prod,
};

static const std::set<StageType> stagesSupportedOutToInPropagation = {
        StageType::Convert,
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    static void validateShapeConversion(const Stage& stage, const Data& shape) {
        const auto& shapeAttrs = shape->attrs();
        VPU_THROW_UNLESS(shapeAttrs.getOrDefault("converted-notation", false),
                "Validation shape conversion while propagation dynamism for stage {} of name {} failed: All shape parent "
                "data object with name {} must be already converted to MDK notation", stage->type(), stage->name(), shape->name());

        const auto& shapeProducer = shape->producer();
        const auto& shapeInIENotation = shapeProducer->input(0);
        VPU_THROW_UNLESS(shapeInIENotation->usage() == DataUsage::Intermediate || shapeInIENotation->usage() == DataUsage::Input,
                "Validation shape conversion while propagation dynamism for stage {} of name {} failed: Shape parent data object (which is "
                "the input with index 0 for shape producer of type {} and name {}) with name {} is expected to be an intermediate or input "
                "data object since shape child is not an output, actual {}",
                stage->type(), stage->name(), shapeProducer->type(), shapeProducer->name(),
                shapeInIENotation->name(), shapeInIENotation->usage());

        const auto& shapeInIENotationAttrs = shapeInIENotation->attrs();
        VPU_THROW_UNLESS(shapeInIENotationAttrs.getOrDefault("IE-notation", false),
                "Validation shape conversion while propagation dynamism for stage {} of name {} failed: Unexpected data object (which is "
                "the input with index 0 for shape producer of type {} and name {}) as shape with name {} in IE notation",
                stage->type(), stage->name(),  shapeProducer->type(), shapeProducer->name(), shapeInIENotation->name());
    }

    static void validateShapes(const Stage& stage, const Data& input, const Data& output) {
        // Propagation is only supported for input and output with equal upper-bound shapes.
        // For example, Prod stage with dynamic input and broadcast are not supported.
        for (const auto& dim : input->desc().dims()) {
            VPU_THROW_UNLESS(dim.second == output->desc().dim(dim.first),
                    "PropagateDynamism: {} stage of name {} must have input of name {} with upper-bound dimension {} "
                    "which should be equal to output which is {}, actual: {}",
                    stage->type(), stage->name(), input->name(), dim.first, output->desc().dim(dim.first), dim.second);
        }
    }

    void propagateFromInputToOutput(
            const Model& model, const Stage& stage,
            const Data& input, const DataToShapeAllocation& parentInputShapeEdge, const Data& output) {
        validateShapes(stage, input, output);

        const auto shape = parentInputShapeEdge->parent();
        validateShapeConversion(stage, shape);

        model->connectDataWithShape(shape, output);

        if (output->usage() == DataUsage::Output) {
            const auto shapeName = createIOShapeName(output->name());
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
    }

    static void propagateFromOutputToInput(
            const Model& model, const Stage& stage,
            const Data& input, const DataToShapeAllocation& parentOutputShapeEdge, const Data& output) {
        validateShapes(stage, input, output);

        const auto shape = parentOutputShapeEdge->parent();
        validateShapeConversion(stage, shape);

        VPU_THROW_UNLESS(input->usage() == DataUsage::Input,
                "PropagateDynamism for stage {} of type {} failed: propagate output dynamism to "
                "input with name {} is available for data with only Input data usage, actual: {}",
                stage->name(), stage->type(), input->name(), shape->usage());

        model->connectDataWithShape(shape, input);
    }

    void run(const Model& model) override {
        VPU_PROFILE(propagateDynamism);

        for (const auto& stage : model->getStages()) {
            if (stagesSupportedInToOutPropagation.count(stage->type())) {
                VPU_THROW_UNLESS(stage->numOutputs() == 1,
                        "PropagateDynamism from input data to output: only single output stages are supported, but {} stage "
                        "of name {} has {} outputs", stage->type(), stage->name(), stage->numOutputs());

                const auto& inputs = stage->inputs();
                std::vector<DataToShapeAllocation> parentInputShapeEdges;

                for (const auto& input : inputs) {
                    if (const auto parentDataToShapeEdge = input->parentDataToShapeEdge()) {
                        parentInputShapeEdges.push_back(parentDataToShapeEdge);
                    }
                }

                const auto& output = stage->output(0);
                const auto& parentOutputShapeEdge = output->parentDataToShapeEdge();

                const auto needPropagateFromInputToOutput = !parentInputShapeEdges.empty() && !parentOutputShapeEdge;

                if (needPropagateFromInputToOutput) {
                    VPU_THROW_UNLESS(parentInputShapeEdges.size() == 1,
                            "PropagateDynamism from input data to output for stage {} of name {} failed: propagation dynamism "
                            "from multiple inputs is not supported, actual number of dynamic inputs: {}",
                            stage->type(), stage->name(), parentInputShapeEdges.size());
                    const auto& parentInputShapeEdge = parentInputShapeEdges[0];
                    const auto& input = parentInputShapeEdge->child();

                    propagateFromInputToOutput(model, stage, input, parentInputShapeEdge, output);
                }
            }

            if (stagesSupportedOutToInPropagation.count(stage->type())) {
                VPU_THROW_UNLESS(stage->numInputs() == 1,
                        "PropagateDynamism from output data to input: only single input stages are supported, but {} stage "
                        "of name {} has {} inputs", stage->type(), stage->name(), stage->numInputs());
                VPU_THROW_UNLESS(stage->numOutputs() == 1,
                        "PropagateDynamism from output data to input: only single output stages are supported, but {} stage "
                        "of name {} has {} outputs", stage->type(), stage->name(), stage->numOutputs());
                const auto& input = stage->input(0);
                const auto& output = stage->output(0);

                const auto& parentInputShapeEdge = input->parentDataToShapeEdge();
                const auto& parentOutputShapeEdge = output->parentDataToShapeEdge();

                const auto needPropagateFromOutputToInput = !parentInputShapeEdge && parentOutputShapeEdge;

                if (needPropagateFromOutputToInput) {
                    propagateFromOutputToInput(model, stage, input, parentOutputShapeEdge, output);
                }
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
