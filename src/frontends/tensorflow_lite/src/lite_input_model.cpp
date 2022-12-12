// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lite_input_model.hpp"

#include <fstream>
#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow_lite {

void InputModel::InputModelTFLiteImpl::loadPlaces() {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::loadPlaces");
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelTFLiteImpl::get_op_places() const {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::get_op_places");
    return {};
}

std::vector<std::shared_ptr<OpPlace>> InputModel::InputModelTFLiteImpl::determine_cut_nodes() const {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::determine_cut_nodes");
    return {};
}

ov::frontend::Place::Ptr InputModel::InputModelTFLiteImpl::getPlaceByTensorName(const std::string& tensorName) const {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::getPlaceByTensorName");
    return nullptr;
}

std::shared_ptr<TensorPlace> castToTensorPlace(const ov::frontend::Place::Ptr& place) {
    FRONT_END_NOT_IMPLEMENTED("InputModelTFLiteImpl; castToTensorPlace");
    return {};
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlaceTF.");
}

void InputModel::InputModelTFLiteImpl::overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::overrideAllInputs");
}

void InputModel::InputModelTFLiteImpl::overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::overrideAllOutputs");
}

void InputModel::InputModelTFLiteImpl::extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                                   const std::vector<ov::frontend::Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::extractSubgraph");
}

void InputModel::InputModelTFLiteImpl::setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape& p_shape) {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::setPartialShape");
}

ov::PartialShape InputModel::InputModelTFLiteImpl::getPartialShape(ov::frontend::Place::Ptr place) const {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::getPartialShape");
}

void InputModel::InputModelTFLiteImpl::setElementType(ov::frontend::Place::Ptr place, const ov::element::Type& type) {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::setElementType");
}

ov::element::Type InputModel::InputModelTFLiteImpl::getElementType(ov::frontend::Place::Ptr place) const {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::getElementType");
    return ov::element::dynamic;
}

void InputModel::InputModelTFLiteImpl::setTensorValue(ov::frontend::Place::Ptr place, const void* value) {
    FRONT_END_NOT_IMPLEMENTED("InputModel::InputModelTFLiteImpl::setTensorValue");
}


InputModel::InputModelTFLiteImpl::InputModelTFLiteImpl(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator,
                                               const ov::frontend::InputModel& input_model)
        : InputModel::InputModelTFImpl(graph_iterator, input_model) {
}

InputModel::InputModelTFLiteImpl::InputModelTFLiteImpl(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator,
                                               const ov::frontend::InputModel& input_model,
                                               const std::shared_ptr<TelemetryExtension>& telemetry)
        : InputModel::InputModelTFImpl(graph_iterator, input_model, telemetry) {
}

InputModel::InputModel(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator, const std::shared_ptr<TelemetryExtension>& telemetry)
        :  ov::frontend::tensorflow::InputModel(std::make_shared<InputModelTFLiteImpl>(graph_iterator, *this, telemetry)) {}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
