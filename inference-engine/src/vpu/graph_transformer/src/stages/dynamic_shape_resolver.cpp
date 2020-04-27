// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

namespace vpu {

void FrontEnd::parseDSR(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 2, "Error while parsing {} with type {}, got {} inputs, while {} were expected",
        layer->name, layer->type, inputs.size(), 2);

    VPU_THROW_UNLESS(outputs.size() == 1, "Error while parsing {} with type {}, got {} outputs, while {} were expected",
                     layer->name, layer->type, outputs.size(), 1);

    const auto& data = inputs[0];
    const auto& shape = inputs[1];

    const auto& dataOutput = outputs[0];

    VPU_THROW_UNLESS(shape->desc().numDims() == 1,
        "Error while parsing {} with type {}, the number of dimensions for the second input {} should be equal to 1 "
        "but got {} instead",
        layer->name, layer->type, shape->name(), shape->desc().numDims());

    VPU_THROW_UNLESS(shape->desc().totalDimSize() == data->desc().numDims(),
        "Error while parsing {} with type {}, the total number of elements for the second input {} should be equal to "
        "the number of dimensions for the first input {}, but got {} and {} respectively",
        layer->name, layer->type, shape->name(), data->name(), shape->desc().totalDimSize(), data->desc().numDims());

    // Dynamic input shape is unsupported
    VPU_THROW_UNLESS(data->producer() != nullptr,
        "Parsing layer {} with type {} failed: DSR stages must have a producer, but actually it doesn't",
        layer->name, layer->type);

    const auto dataOutputEdge = data->producerEdge();
    const auto shapeOutputEdge = shape->producerEdge();

    if (dataOutput->usage() == DataUsage::Output) {
        // Create the second output with shape in case of dynamic output
        const auto& shapeOutput = model->addOutputData(dataOutput->name() + "@shape", shape->desc());

        model->replaceStageOutput(shapeOutputEdge, shapeOutput);

        model->removeUnusedData(shape);
    } else {
        model->connectDataWithShape(shape, dataOutput);
    }

    model->replaceStageOutput(dataOutputEdge, dataOutput);
    model->removeUnusedData(data);
}

}  // namespace vpu
