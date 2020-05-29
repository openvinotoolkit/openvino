// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

namespace vpu {

void FrontEnd::parseDSR(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 2, "Error while parsing {} of type {}, got {} inputs, while {} were expected",
        layer->name, layer->type, inputs.size(), 2);
    const auto& data = inputs[0];
    const auto& shape = inputs[1];

    VPU_THROW_UNLESS(outputs.size() == 1, "Parsing layer {} of type {} failed: got {} outputs, while {} were expected",
         layer->name, layer->type, outputs.size(), 1);
    const auto& dataOutput = outputs[0];

    const auto dataProducerEdge = data->producerEdge();
    VPU_THROW_UNLESS(dataProducerEdge != nullptr, "Parsing layer {} of type {} failed: input with index {} (of name {}) must have a producer",
        layer->name, layer->type, 0, data->name());

    VPU_THROW_UNLESS(shape->desc().numDims() == 1,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have rank equal to {}, actual is {}",
        layer->name, layer->type, 0, shape->name(), 1, shape->desc().numDims());

    VPU_THROW_UNLESS(shape->desc().totalDimSize() == data->desc().numDims(),
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have the same total elements number as "
        "input with index {} (of name {}), actual {} and {} respectively",
        layer->name, layer->type, 0, shape->name(), 1, data->name(), shape->desc().totalDimSize(), data->desc().numDims());

    const auto shapeProducerEdge = shape->producerEdge();
    VPU_THROW_UNLESS(shapeProducerEdge != nullptr, "Parsing layer {} of type {} failed: input with index {} (of name {}) must have a producer",
        layer->name, layer->type, 1, shape->name());

    if (auto dataToShapeEdge = data->parentDataToShapeEdge()) {
        const auto& parent = dataToShapeEdge->parent();
        VPU_THROW_UNLESS(parent == shape, "Myriad plugin encountered layer of type \"{}\" and name \"{}\" with input #{} (data input with name \"{}\") that "
            "already has parent in terms of data to shape connection. The parent is expected to be input #{} (shape input with name \"{}\") of the layer, so "
            "it's a \"{}\" with already connected inputs, but actual parent is other data object with name \"{}\". The case of connected inputs is considered "
            "as \"{}\" that goes directly to \"{}\" as a result of some optimization (operation between them has been optimized out). Other cases, when some "
            "input already has a connection, but with other data object are prohibited.",
            layer->type, layer->name, 0, data->name(), 1, shape->name(), layer->type, parent->name(), layer->type, layer->type);
        model->disconnectDatas(dataToShapeEdge);
    }
    model->replaceStageOutput(dataProducerEdge, dataOutput);
    model->removeUnusedData(data);

    if (dataOutput->usage() == DataUsage::Output) {
        // Create the second output with shape in case of dynamic output
        const auto& shapeOutput = model->addOutputData(dataOutput->name() + "@shape", shape->desc());

        model->replaceStageOutput(shapeProducerEdge, shapeOutput);
        model->connectDataWithShape(shapeOutput, dataOutput);

        for (const auto& dataToShapeEdge : shape->childDataToShapeEdges()) {
            model->replaceDataToShapeParent(dataToShapeEdge, shapeOutput);
        }

        model->removeUnusedData(shape);
    } else {
        model->connectDataWithShape(shape, dataOutput);
    }
}

}  // namespace vpu
