// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <vpu/utils/shape_io.hpp>

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
namespace vpu {

void FrontEnd::parseDSR(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) {
    auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(node);
    VPU_THROW_UNLESS(inputs.size() == 2, "Error while parsing {} of type {}, got {} inputs, while {} were expected",
        dsr->get_name(), dsr->get_type_name(), inputs.size(), 2);
    const auto& data = inputs[0];
    const auto& shape = inputs[1];

    VPU_THROW_UNLESS(outputs.size() == 1, "Parsing layer {} of type {} failed: got {} outputs, while {} were expected",
         dsr->get_name(), dsr->get_type_name(), outputs.size(), 1);
    auto dataOutput = outputs[0];

    VPU_THROW_UNLESS(!dsr || dsr->get_input_source_output(0).get_target_inputs().size() == 1,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must not be an input for any operation except current "
        "of type {}, actual number of operations for which data is input is {}. "
        "DynamicToStaticShape transformations should add {} operation after all operations with dynamic output as only "
        "consumer. All operations that were previously original output data consumers should now consume the output data "
        "from {}. Otherwise the consumer which was not redirected to {} output would process garbage data.",
        dsr->get_name(), dsr->get_type_name(), 0, data->name(), dsr->get_type_name(), dsr->get_input_source_output(0).get_target_inputs().size(),
        dsr->get_name(), dsr->get_type_name());

    VPU_THROW_UNLESS(data->consumerEdges().size() == 0,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have no consumers, actual: {}. "
        "DynamicToStaticShape transformations should add {} operation after all operations with dynamic output as only "
        "consumer. All operations that were previously original output data consumers should now consume the output data "
        "from {}. Otherwise the consumer which was not redirected to {} output would process garbage data.",
        dsr->get_name(), dsr->get_type_name(), 0, data->name(), data->consumerEdges().size(), dsr->get_type_name(), dsr->get_type_name(), dsr->get_type_name());

    VPU_THROW_UNLESS(shape->desc().numDims() == 1,
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have rank equal to {}, actual is {}",
        dsr->get_name(), dsr->get_type_name(), 0, shape->name(), 1, shape->desc().numDims());

    VPU_THROW_UNLESS(shape->desc().totalDimSize() == data->desc().numDims(),
        "Parsing layer {} of type {} failed: input with index {} (of name {}) must have the same total elements number as "
        "input with index {} (of name {}), actual {} and {} respectively",
        dsr->get_name(), dsr->get_type_name(), 0, shape->name(), 1, data->name(), shape->desc().totalDimSize(), data->desc().numDims());

    const auto dataProducerEdge = data->producerEdge();
    const auto shapeProducerEdge = shape->producerEdge();

    if (dataProducerEdge == nullptr) {
        VPU_THROW_UNLESS(data->usage() == DataUsage::Input,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has not a producer, it must have Input "
            "data usage, actual: {}", dsr->get_name(), dsr->get_type_name(), 0, data->name(), data->usage());
        const auto& origNode = dataOutput->origNode();
        VPU_THROW_UNLESS(origNode != nullptr,
            "Parsing layer {} of type {} failed: output data with index {} (of name {}) must have original IE data",
            dsr->get_name(), dsr->get_type_name(), 0, dataOutput->name());

        bindData(data, origNode->get_input_source_output(0), origNode);
        model->removeUnusedData(dataOutput);
        dataOutput = data;
    } else {
        VPU_THROW_UNLESS(data->usage() == DataUsage::Intermediate,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has a producer, it must have Intermediate "
            "data usage, actual: {}", dsr->get_name(), dsr->get_type_name(), 0, data->name(), data->usage());

        if (auto dataToShapeEdge = data->parentDataToShapeEdge()) {
            const auto& parent = dataToShapeEdge->parent();
            VPU_THROW_UNLESS(parent == shape,
                "Myriad plugin encountered layer of type \"{}\" and name \"{}\" with input #{} (data input with name \"{}\") that "
                "already has parent in terms of data to shape connection. The parent is expected to be input #{} (shape input with "
                "name \"{}\") of the layer, so it's a \"{}\" with already connected inputs, but actual parent is other data object "
                "with name \"{}\". The case of connected inputs is considered as \"{}\" that goes directly to \"{}\" as a result of "
                "some optimization (operation between them has been optimized out). Other cases, when some input already has a "
                "connection, but with other data object are prohibited.",
                dsr->get_name(), dsr->get_type_name(), 0, data->name(), 1, shape->name(),
                 dsr->get_type_name(), parent->name(), dsr->get_type_name(), dsr->get_type_name());
            model->disconnectDatas(dataToShapeEdge);
        }
        model->replaceStageOutput(dataProducerEdge, dataOutput);
        model->removeUnusedData(data);
    }

    if (shapeProducerEdge == nullptr) {
        VPU_THROW_UNLESS(shape->usage() == DataUsage::Input,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has not a producer, it must have Input "
            "data usage, actual: {}", dsr->get_name(), dsr->get_type_name(), 1, shape->name(), shape->usage());
    } else {
        VPU_THROW_UNLESS(shape->usage() == DataUsage::Intermediate || shape->usage() == DataUsage::Output,
            "Parsing layer {} of type {} failed: if input with index {} (of name {}) has a producer, it must have Intermediate "
            "or Output (if already has been associated with other output data) data usage, actual: {}",
            dsr->get_name(), dsr->get_type_name(), 1, shape->name(), shape->usage());
    }

    auto shapeDataObject = shape;
    if (dataOutput->usage() == DataUsage::Output && shapeDataObject->usage() != DataUsage::Output) {
        const auto& shapeOutput = model->addOutputData(createIOShapeName(dataOutput->name()), shape->desc());

        bindData(shapeOutput, shape->origNode()->get_input_source_output(0), shape->origNode());
        for (const auto& shapeConsumerEdge : shape->consumerEdges()) {
            model->replaceStageInput(shapeConsumerEdge, shapeOutput);
        }

        for (const auto& dataToShapeEdge : shape->childDataToShapeEdges()) {
            model->replaceDataToShapeParent(dataToShapeEdge, shapeOutput);
        }

        if (!shapeProducerEdge) {
            _stageBuilder->addCopyStage(
                    model,
                    dsr->get_name() + "@copy-for-dynamic-output",
                    dsr,
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
