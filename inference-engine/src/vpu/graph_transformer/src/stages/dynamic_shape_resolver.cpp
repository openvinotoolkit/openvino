// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <ngraph/node.hpp>

namespace vpu {

void FrontEnd::parseDSR(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) {
    VPU_THROW_UNLESS(inputs.size() == 2, "Error while parsing {} of type {}, got {} inputs, while {} were expected",
        layer->name, layer->type, inputs.size(), 2);
    const auto& data = inputs[0];
    const auto& shape = inputs[1];

    VPU_THROW_UNLESS(outputs.size() == 1, "Parsing layer {} of type {} failed: got {} outputs, while {} were expected",
         layer->name, layer->type, outputs.size(), 1);
    auto dataOutput = outputs[0];

    const auto ngraphNode = layer->getNode();
    VPU_THROW_UNLESS(!ngraphNode || ngraphNode->get_input_source_output(0).get_target_inputs().size() == 1,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must not be an input for any operation except current "
        "of type {}, actual number of operations for which data is input is {}. "
        "DynamicToStaticShape transformations should add {} operation after all operations with dynamic output as only "
        "consumer. All operations that were previously original output data consumers should now consume the output data "
        "from {}. Otherwise the consumer which was not redirected to {} output would process garbage data.",
        layer->name, layer->type, 0, data->name(), layer->type, ngraphNode->get_input_source_output(0).get_target_inputs().size(),
        layer->type, layer->type);
    VPU_THROW_UNLESS(data->consumerEdges().size() == 0,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have no consumers, actual: {}. "
        "DynamicToStaticShape transformations should add {} operation after all operations with dynamic output as only "
        "consumer. All operations that were previously original output data consumers should now consume the output data "
        "from {}. Otherwise the consumer which was not redirected to {} output would process garbage data.",
        layer->name, layer->type, 0, data->name(), data->consumerEdges().size(), layer->type, layer->type, layer->type);

    VPU_THROW_UNLESS(shape->desc().numDims() == 1,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have rank equal to {}, actual is {}",
        layer->name, layer->type, 0, shape->name(), 1, shape->desc().numDims());

    VPU_THROW_UNLESS(shape->desc().totalDimSize() == data->desc().numDims(),
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have the same total elements number as "
        "input with index {} (of name {}), actual {} and {} respectively",
        layer->name, layer->type, 0, shape->name(), 1, data->name(), shape->desc().totalDimSize(), data->desc().numDims());

    const auto dataProducerEdge = data->producerEdge();
    const auto shapeProducerEdge = shape->producerEdge();

    if (dataProducerEdge == nullptr) {
        VPU_THROW_UNLESS(data->usage() == DataUsage::Input,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has not a producer, it must have Input "
            "data usage, actual: {}", layer->name, layer->type, 0, data->name(), data->usage());
        const auto& origData = dataOutput->origData();
        VPU_THROW_UNLESS(origData != nullptr,
            "Parsing layer {} of type {} failed: output data with index {} (of name {}) must have original IE data",
            layer->name, layer->type, 0, dataOutput->name());

        bindData(data, origData);
        model->removeUnusedData(dataOutput);
        dataOutput = data;
    } else {
        VPU_THROW_UNLESS(data->usage() == DataUsage::Intermediate,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has a producer, it must have Intermediate "
            "data usage, actual: {}", layer->name, layer->type, 0, data->name(), data->usage());

        if (auto dataToShapeEdge = data->parentDataToShapeEdge()) {
            const auto& parent = dataToShapeEdge->parent();
            VPU_THROW_UNLESS(parent == shape,
                "Myriad plugin encountered layer of type \"{}\" and name \"{}\" with input #{} (data input with name \"{}\") that "
                "already has parent in terms of data to shape connection. The parent is expected to be input #{} (shape input with "
                "name \"{}\") of the layer, so it's a \"{}\" with already connected inputs, but actual parent is other data object "
                "with name \"{}\". The case of connected inputs is considered as \"{}\" that goes directly to \"{}\" as a result of "
                "some optimization (operation between them has been optimized out). Other cases, when some input already has a "
                "connection, but with other data object are prohibited.",
                layer->type, layer->name, 0, data->name(), 1, shape->name(),
                layer->type, parent->name(), layer->type, layer->type);
            model->disconnectDatas(dataToShapeEdge);
        }
        model->replaceStageOutput(dataProducerEdge, dataOutput);
        model->removeUnusedData(data);
    }

    if (shapeProducerEdge == nullptr) {
        VPU_THROW_UNLESS(shape->usage() == DataUsage::Input,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has not a producer, it must have Input "
            "data usage, actual: {}", layer->name, layer->type, 1, shape->name(), shape->usage());
    } else {
        VPU_THROW_UNLESS(shape->usage() == DataUsage::Intermediate || shape->usage() == DataUsage::Output,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has a producer, it must have Intermediate "
            "or Output (if already has been associated with other output data) data usage, actual: {}",
            layer->name, layer->type, 1, shape->name(), shape->usage());
    }

    auto shapeDataObject = shape;
    if (dataOutput->usage() == DataUsage::Output && shapeDataObject->usage() != DataUsage::Output) {
        const auto& shapeOutput = model->addOutputData(dataOutput->name() + "@shape", shape->desc());

        bindData(shapeOutput, shape->origData());
        for (const auto& shapeConsumerEdge : shape->consumerEdges()) {
            model->replaceStageInput(shapeConsumerEdge, shapeOutput);
        }

        for (const auto& dataToShapeEdge : shape->childDataToShapeEdges()) {
            model->replaceDataToShapeParent(dataToShapeEdge, shapeOutput);
        }

        if (!shapeProducerEdge) {
            _stageBuilder->addCopyStage(
                    model,
                    layer->name + "@copy-for-dynamic-output",
                    layer,
                    shape,
                    shapeOutput,
                    "DynamicShapeResolver");
        } else {
            model->replaceStageOutput(shapeProducerEdge, shapeOutput);
            model->removeUnusedData(shape);
        }

        shapeDataObject = shapeOutput;
    }
    model->connectDataWithShape(shapeDataObject, dataOutput);
}

}  // namespace vpu
