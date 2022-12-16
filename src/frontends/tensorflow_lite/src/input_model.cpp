// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <fstream>
#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class InputModel::InputModelTFLiteImpl {
public:
    InputModelTFLiteImpl(const GraphIteratorFlatBuffer::Ptr& graph_iterator, const ov::frontend::InputModel& input_model);
    InputModelTFLiteImpl(const GraphIteratorFlatBuffer::Ptr& graph_iterator,
                         const ov::frontend::InputModel& input_model,
                         const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<ov::frontend::Place::Ptr> getInputs() const;
    std::vector<ov::frontend::Place::Ptr> getOutputs() const;
//    ov::frontend::Place::Ptr getPlaceByTensorName(const std::string& tensorName) const;
//    void overrideAllOutputs(const std::vector<ov::frontend::Place::Ptr>& outputs);
//    void overrideAllInputs(const std::vector<ov::frontend::Place::Ptr>& inputs);
//    void extractSubgraph(const std::vector<ov::frontend::Place::Ptr>& inputs,
//                         const std::vector<ov::frontend::Place::Ptr>& outputs);
//    void setPartialShape(ov::frontend::Place::Ptr place, const ov::PartialShape&);
//    ov::PartialShape getPartialShape(ov::frontend::Place::Ptr place) const;
//    void setElementType(ov::frontend::Place::Ptr place, const ov::element::Type&);
//    ov::element::Type getElementType(ov::frontend::Place::Ptr place) const;
//    void setTensorValue(ov::frontend::Place::Ptr place, const void* value);

    std::vector<std::shared_ptr<OpPlace>> get_op_places() const {
        return m_op_places;
    }
    std::map<std::string, std::shared_ptr<TensorPlace>> get_tensor_places() const {
        return m_tensor_places;
    }
    std::map<std::string, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    }

private:
    void loadModel();

    std::vector<std::shared_ptr<OpPlace>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlace>> m_op_places_map;
    std::map<std::string, std::shared_ptr<TensorPlace>> m_tensor_places;
    std::vector<ov::frontend::Place::Ptr> m_inputs;
    std::vector<ov::frontend::Place::Ptr> m_outputs;
    std::map<std::string, Output<Node>> m_tensor_values;

    std::shared_ptr<GraphIteratorFlatBuffer> m_graph_iterator;
    const ov::frontend::InputModel& m_input_model;

    std::shared_ptr<TelemetryExtension> m_telemetry;
};


void InputModel::InputModelTFLiteImpl::loadModel() {
    std::unordered_set<size_t> non_constant_tensors;

    // inputs
    const auto& input_tensor_indices = m_graph_iterator->get_model_input_tensor_indices();
    m_inputs.reserve(input_tensor_indices.size());
    for (const auto& i : input_tensor_indices) {
        auto tensor = m_graph_iterator->get_tensor(i);
        const auto& names = std::vector<std::string>{tensor->name()->str()};
        const auto& ov_shape = get_ov_shape(tensor->shape());
        const auto& ov_type = get_ov_type(tensor->type());
        m_inputs.push_back(std::make_shared<ov::frontend::tensorflow::TensorPlace>(m_input_model, ov_shape, ov_type, names));
        non_constant_tensors.insert(i);
    }

    // outputs
    const auto& output_tensor_indices = m_graph_iterator->get_model_output_tensor_indices();
    m_outputs.reserve(output_tensor_indices.size());
    for (const auto& i : output_tensor_indices) {
        auto tensor = m_graph_iterator->get_tensor(i);
        const auto& name = tensor->name()->str();
        const auto& ov_shape = get_ov_shape(tensor->shape());
        const auto& ov_type = get_ov_type(tensor->type());
        m_outputs.push_back(std::make_shared<ov::frontend::tensorflow::TensorPlace>(m_input_model, ov_shape, ov_type, std::vector<std::string>{name}));
        non_constant_tensors.insert(i);
    }

    // ops
    m_op_places.reserve(m_graph_iterator->size());
    for (; !m_graph_iterator->is_end(); m_graph_iterator->next()) {
        const auto& decoder = m_graph_iterator->get_decoder();
        m_op_places.push_back(std::make_shared<OpPlace>(m_input_model, decoder));
        for (size_t idx : decoder->get_output_tensor_indices())
            non_constant_tensors.insert(idx);
    }

    // constant data aka buffer data
    const auto tensors = m_graph_iterator->get_tensors();
    const auto buffers = m_graph_iterator->get_buffers();
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto tensor = tensors[i];
        const auto& name = tensor->name()->str();
        const auto& type = get_ov_type(tensor->type());
        const auto& shape = get_ov_shape(tensor->shape()).to_shape();
        if (non_constant_tensors.find(i) == non_constant_tensors.end()) {
            auto buffer = buffers[tensor->buffer()]->data()->data();
            auto constant = ov::op::v0::Constant::create(type, shape, buffer);
            m_tensor_values[name] = constant;
        }
        m_tensor_places[name] = std::make_shared<TensorPlace>(m_input_model, shape, type, std::vector<std::string>{name});
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

InputModel::InputModel(const GraphIteratorFlatBuffer::Ptr& graph_iterator, const std::shared_ptr<TelemetryExtension>& telemetry) : _impl{std::make_shared<InputModelTFLiteImpl>(graph_iterator, *this, telemetry)} {}

std::vector<std::shared_ptr<ov::frontend::tensorflow::OpPlace>> InputModel::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<ov::frontend::tensorflow::TensorPlace>>
InputModel::get_tensor_places() const {
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

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
