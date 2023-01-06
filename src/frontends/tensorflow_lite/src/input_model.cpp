// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/util/log.hpp"
#include "tensor_lite_place.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class InputModel::InputModelTFLiteImpl {
public:
    InputModelTFLiteImpl(const GraphIteratorFlatBuffer::Ptr& graph_iterator,
                         const ov::frontend::InputModel& input_model);
    InputModelTFLiteImpl(const GraphIteratorFlatBuffer::Ptr& graph_iterator,
                         const ov::frontend::InputModel& input_model,
                         const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<ov::frontend::Place::Ptr> getInputs() const;
    std::vector<ov::frontend::Place::Ptr> getOutputs() const;
    ov::frontend::Place::Ptr getPlaceByTensorName(const std::string& tensorName) const;

    /////  Searching for places  /////
    std::vector<std::shared_ptr<OpPlace>> get_op_places() const {
        return m_op_places;
    }
    std::map<std::string, std::shared_ptr<TensorLitePlace>> get_tensor_places() const {
        return m_tensor_places;
    }
    std::map<std::string, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    }

    ///// Naming and annotation  /////
    void setNameForTensor(const Place::Ptr& tensor, const std::string& new_name);
    void addNameForTensor(const Place::Ptr& tensor, const std::string& new_name);
    void setNameForOperation(const Place::Ptr& operation, const std::string& new_name);

    ///// Setting / getting tensor properties  /////
    void setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape& shape);
    ov::PartialShape getPartialShape(ov::frontend::Place::Ptr place) const;
    void setElementType(ov::frontend::Place::Ptr place, const ov::element::Type& type);
    ov::element::Type getElementType(ov::frontend::Place::Ptr place) const;
    void setTensorValue(ov::frontend::Place::Ptr place, const void* value);

    ///// Topology Editing  /////
    void overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
    void overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
    void extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                         const std::vector<ov::frontend::Place::Ptr>& outputs);

private:
    void loadModel();
    void cleanUp();

    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
    std::map<std::string, std::shared_ptr<TensorLitePlace>> m_tensor_places;
    std::vector<ov::frontend::Place::Ptr> m_inputs;
    std::vector<ov::frontend::Place::Ptr> m_outputs;
    std::map<std::string, Output<Node>> m_tensor_values;

    std::shared_ptr<GraphIteratorFlatBuffer> m_graph_iterator;
    const ov::frontend::InputModel& m_input_model;

    std::shared_ptr<TelemetryExtension> m_telemetry;
};

void InputModel::InputModelTFLiteImpl::loadModel() {
    std::map<std::string, uint64_t> op_statistics;  // for telemetry

    m_op_places.reserve(m_graph_iterator->size());
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        const auto& decoder = m_graph_iterator->get_decoder();
        m_op_places.push_back(std::make_shared<OpPlace>(m_input_model, decoder));

        if (m_telemetry) {
            op_statistics[decoder->get_op_type()]++;
        }

        for (size_t i = 0; i < decoder->get_input_size(); ++i) {
            auto place = decoder->decode_input_tensor(i, m_input_model);
            auto name = place->get_names()[0];
            if (m_tensor_places.find(name) == m_tensor_places.end()) {
                m_tensor_places[name] = place;
                if (place->is_input()) {
                    // will reorder by index later
                    m_inputs.push_back(place);
                } else if (auto data = place->get_data()) {
                    auto constant = ov::op::v0::Constant::create(place->get_element_type(),
                                                                 place->get_partial_shape().to_shape(),
                                                                 data);
                    m_tensor_values[name] = constant;
                } else {
                    FRONT_END_GENERAL_CHECK(false,
                                            "This tensor should be either input, constant or ",
                                            "should be already produced by previous operators: ",
                                            name,
                                            ". Error is encountered while working with operation of type ",
                                            decoder->get_op_type(),
                                            " and name ",
                                            decoder->get_op_name(),
                                            ".");
                }
            }
        }
        for (size_t i = 0; i < decoder->get_output_size(); ++i) {
            auto place = decoder->decode_output_tensor(i, m_input_model);
            auto name = place->get_names()[0];
            if (m_tensor_places.find(name) == m_tensor_places.end()) {
                m_tensor_places[name] = place;
                if (place->is_output()) {
                    // will reorder by index later
                    m_outputs.push_back(place);
                }
            }
        }
    }

    auto sorting_places_by_idx = [](bool are_input_places) {
        return
            [are_input_places](const ov::frontend::Place::Ptr& lhs_place, const ov::frontend::Place::Ptr& rhs_place) {
                auto tflite_lhs_place =
                    std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(lhs_place);
                auto tflite_rhs_place =
                    std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(rhs_place);
                FRONT_END_GENERAL_CHECK(tflite_lhs_place != nullptr && tflite_rhs_place != nullptr,
                                        "TFLite Frontend works with TensorLitePlaces only");
                size_t rhs_idx, lhs_idx;
                if (are_input_places) {
                    lhs_idx = tflite_lhs_place->get_input_index();
                    rhs_idx = tflite_rhs_place->get_input_index();
                } else {
                    lhs_idx = tflite_lhs_place->get_output_index();
                    rhs_idx = tflite_rhs_place->get_output_index();
                }
                return lhs_idx < rhs_idx;
            };
    };
    std::sort(m_inputs.begin(), m_inputs.end(), sorting_places_by_idx(true));
    std::sort(m_outputs.begin(), m_outputs.end(), sorting_places_by_idx(false));

    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "tflite_" + op.first, static_cast<int>(op.second));
        }
    }
}

InputModel::InputModelTFLiteImpl::InputModelTFLiteImpl(const GraphIteratorFlatBuffer::Ptr& graph_iterator,
                                                       const ov::frontend::InputModel& input_model)
    : m_input_model(input_model),
      m_graph_iterator(graph_iterator) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    loadModel();
}

InputModel::InputModelTFLiteImpl::InputModelTFLiteImpl(const GraphIteratorFlatBuffer::Ptr& graph_iterator,
                                                       const ov::frontend::InputModel& input_model,
                                                       const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_input_model(input_model),
      m_graph_iterator(graph_iterator),
      m_telemetry(telemetry) {
    FRONT_END_GENERAL_CHECK(m_graph_iterator, "Null pointer specified for GraphIterator");
    loadModel();
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFLiteImpl::getInputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::InputModelTFLiteImpl::getOutputs() const {
    return m_outputs;
}

std::shared_ptr<TensorPlace> castToTensorPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlace>(place)) {
        return var_place;
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlace.");
}

ov::frontend::Place::Ptr InputModel::InputModelTFLiteImpl::getPlaceByTensorName(const std::string& tensorName) const {
    if (m_tensor_places.find(tensorName) != m_tensor_places.end())
        return castToTensorPlace(m_tensor_places.at(tensorName));
    else
        return nullptr;
}

std::shared_ptr<OpPlace> castToOpPlace(const ov::frontend::Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<OpPlace>(place)) {
        return var_place;
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlace.");
}

void InputModel::InputModelTFLiteImpl::setPartialShape(ov::frontend::Place::Ptr place, const PartialShape& shape) {
    castToTensorPlace(place)->set_partial_shape(shape);
}

ov::PartialShape InputModel::InputModelTFLiteImpl::getPartialShape(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void InputModel::InputModelTFLiteImpl::setElementType(ov::frontend::Place::Ptr place, const element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

ov::element::Type InputModel::InputModelTFLiteImpl::getElementType(ov::frontend::Place::Ptr place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::InputModelTFLiteImpl::setTensorValue(ov::frontend::Place::Ptr place, const void* value) {
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    FRONT_END_GENERAL_CHECK(tensor_place->get_names().size() > 0,
                            "TensorFlow Lite Frontend: place to be frozen must have the name.");
    auto name = tensor_place->get_names()[0];
    FRONT_END_GENERAL_CHECK(p_shape.is_static(),
                            "TensorFlow Lite Frontend: specify static shape for " + name + " to be frozen.");
    FRONT_END_GENERAL_CHECK(type.is_static(),
                            "TensorFlow Lite Frontend: define static size type for " + name + " to be frozen.");
    auto constant = opset10::Constant::create(type, p_shape.to_shape(), value);
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

void InputModel::InputModelTFLiteImpl::setNameForTensor(const Place::Ptr& tensor, const std::string& new_name) {
    castToTensorPlace(tensor)->set_names({new_name});
}

void InputModel::InputModelTFLiteImpl::addNameForTensor(const Place::Ptr& tensor, const std::string& new_name) {
    auto tf_tensor = castToTensorPlace(tensor);
    auto names = tf_tensor->get_names();
    names.push_back(new_name);
    tf_tensor->set_names(names);
}

void InputModel::InputModelTFLiteImpl::setNameForOperation(const Place::Ptr& operation, const std::string& new_name) {
    auto op = castToOpPlace(operation);
    auto names = op->get_names();
    names.push_back(new_name);
    op->set_names(names);
}

void InputModel::InputModelTFLiteImpl::overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    for (const auto& input_place : m_inputs) {
        auto input_lite_place = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(input_place);
        FRONT_END_GENERAL_CHECK(input_lite_place != nullptr, "");  // FIXME
        input_lite_place->set_input_index(-1);
    }
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
    cleanUp();
}

void InputModel::InputModelTFLiteImpl::overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    for (const auto& output_place : m_outputs) {
        auto output_lite_place =
            std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "");  // FIXME
        output_lite_place->set_output_index(-1);
    }
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
    cleanUp();
}

void InputModel::InputModelTFLiteImpl::extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                                       const std::vector<ov::frontend::Place::Ptr>& outputs) {
    for (const auto& input_place : m_inputs) {
        auto input_lite_place = std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(input_place);
        FRONT_END_GENERAL_CHECK(input_lite_place != nullptr, "");  // FIXME
        input_lite_place->set_input_index(-1);
    }
    m_inputs.clear();
    for (const auto& input_place : inputs) {
        m_inputs.push_back(castToTensorPlace(input_place));
    }
    for (const auto& output_place : m_outputs) {
        auto output_lite_place =
            std::dynamic_pointer_cast<ov::frontend::tensorflow_lite::TensorLitePlace>(output_place);
        FRONT_END_GENERAL_CHECK(output_lite_place != nullptr, "");  // FIXME
        output_lite_place->set_output_index(-1);
    }
    m_outputs.clear();
    for (const auto& output_place : outputs) {
        m_outputs.push_back(castToTensorPlace(output_place));
    }
    cleanUp();
}

void InputModel::InputModelTFLiteImpl::cleanUp() {
    // TODO: remove all the unnecessary tensors and operations now!
}

InputModel::InputModel(const GraphIteratorFlatBuffer::Ptr& graph_iterator,
                       const std::shared_ptr<TelemetryExtension>& telemetry)
    : _impl{std::make_shared<InputModelTFLiteImpl>(graph_iterator, *this, telemetry)} {}

std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> InputModel::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace>> InputModel::get_tensor_places()
    const {
    return _impl->get_tensor_places();
}

std::map<std::string, Output<Node>> InputModel::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return _impl->getInputs();
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return _impl->getOutputs();
}

ov::frontend::Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->getPlaceByTensorName(tensorName);
}

void InputModel::set_partial_shape(const Place::Ptr& place, const PartialShape& shape) {
    _impl->setPartialShape(place, shape);
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    return _impl->getPartialShape(place);
}

void InputModel::set_element_type(const Place::Ptr& place, const element::Type& type) {
    _impl->setElementType(place, type);
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    return _impl->getElementType(place);
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    _impl->setTensorValue(place, value);
}

void InputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    _impl->setNameForTensor(tensor, new_name);
}

void InputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    _impl->addNameForTensor(tensor, new_name);
}

void InputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    _impl->setNameForOperation(operation, new_name);
}

void InputModel::override_all_outputs(const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->overrideAllOutputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<ov::frontend::Place::Ptr>& inputs) {
    _impl->overrideAllInputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
                                  const std::vector<ov::frontend::Place::Ptr>& outputs) {
    _impl->extractSubgraph(inputs, outputs);
}

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
