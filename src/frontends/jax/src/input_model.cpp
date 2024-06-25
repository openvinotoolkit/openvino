// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "openvino/frontend/exception.hpp"
#include "place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {

InputModel::InputModel(const std::shared_ptr<JaxDecoder>& model_decoder) : m_model_decoder(model_decoder) {
    const auto& inputs = m_model_decoder->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto in_place = std::make_shared<jax::Place>(*this, inputs[i]);
        m_name_to_place.emplace(std::to_string(inputs[i]), std::dynamic_pointer_cast<frontend::Place>(in_place));
        for (const auto& name : in_place->get_names()) {
            m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(in_place));
        }
        m_inputs.push_back(in_place);
    }
    const auto& outputs = m_model_decoder->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto out_place = std::make_shared<jax::Place>(*this, outputs[i]);
        m_name_to_place.emplace(std::to_string(outputs[i]), std::dynamic_pointer_cast<frontend::Place>(out_place));
        for (const auto& name : out_place->get_names()) {
            m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(out_place));
        }
        m_outputs.push_back(out_place);
    }
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return m_outputs;
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    auto place_it = m_name_to_place.find(tensor_name);
    if (place_it != m_name_to_place.end()) {
        return place_it->second;
    }
    return nullptr;
}

void InputModel::set_partial_shape(const Place::Ptr& place, const ov::PartialShape& shape) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting shape.");
    auto jax_place = std::dynamic_pointer_cast<jax::Place>(place);
    FRONT_END_GENERAL_CHECK(jax_place, "Only place produced by Jax Frontend is supported");
    jax_place->m_pshape = shape;
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    auto jax_place = std::dynamic_pointer_cast<jax::Place>(place);
    FRONT_END_GENERAL_CHECK(jax_place,
                            "Provided place is invalid. Only place of input or output is supported by Jax Frontend.");
    return jax_place->get_partial_shape();
}

void InputModel::set_element_type(const Place::Ptr& place, const ov::element::Type& type) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting element type.");
    auto jax_place = std::dynamic_pointer_cast<jax::Place>(place);
    FRONT_END_GENERAL_CHECK(jax_place, "Only place produced by Jax Frontend is supported");
    jax_place->m_type = type;
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    auto jax_place = std::dynamic_pointer_cast<jax::Place>(place);
    FRONT_END_GENERAL_CHECK(jax_place,
                            "Provided place is invalid. Only place of input or output is supported by Jax Frontend.");
    return jax_place->get_element_type();
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    FRONT_END_NOT_IMPLEMENTED("set_tensor_value");
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    // Topology modification is not allowed, but order can be changed
    auto all_outputs = std::all_of(outputs.cbegin(), outputs.cend(), [](const Place::Ptr& p) {
        return p->is_output();
    });
    FRONT_END_GENERAL_CHECK(all_outputs, "Only initial outputs are supported by override_all_inputs.");
    // Number of outputs can be lower then in original model
    m_outputs = outputs;
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    // Topology modification is not allowed, but order can be changed
    auto all_inputs = std::all_of(inputs.cbegin(), inputs.cend(), [](const Place::Ptr& p) {
        return p->is_input();
    });
    FRONT_END_GENERAL_CHECK(all_inputs, "Only initial inputs are supported by override_all_inputs.");
    FRONT_END_GENERAL_CHECK(inputs.size() == m_inputs.size(),
                            "Number of inputs provided is incorrect. Graph modification is not supported for "
                            "this model. Expected number of inputs: ",
                            m_inputs.size(),
                            " received ",
                            inputs.size());
    m_inputs = inputs;
}

std::shared_ptr<JaxDecoder> InputModel::get_decoder() const {
    return m_model_decoder;
}

}  // namespace jax
}  // namespace frontend
}  // namespace ov
