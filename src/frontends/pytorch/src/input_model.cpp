// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

InputModel::InputModel(const std::shared_ptr<TorchDecoder>& model_decoder)
    : m_model_decoder(model_decoder),
      m_decoder_type_name(model_decoder->decoder_type_name()) {
    const auto& inputs = m_model_decoder->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto in_place = std::make_shared<pytorch::Place>(*this, inputs[i]);
        m_name_to_place.emplace(std::to_string(inputs[i]), std::dynamic_pointer_cast<frontend::Place>(in_place));
        for (const auto& name : in_place->get_names()) {
            m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(in_place));
        }
        m_inputs.push_back(in_place);
    }
    const auto& outputs = m_model_decoder->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto out_place = std::make_shared<pytorch::Place>(*this, outputs[i]);
        m_name_to_place.emplace(std::to_string(outputs[i]), std::dynamic_pointer_cast<frontend::Place>(out_place));
        for (const auto& name : out_place->get_names()) {
            m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(out_place));
        }
        m_outputs.push_back(out_place);
    }
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    if (m_inputs.size() > 0 && m_inputs[0]) {
        // We need to remove "self" input to not confuse external users
        const auto& names = m_inputs[0]->get_names();
        if (std::any_of(names.cbegin(), names.cend(), [](const std::string& n) {
                return n.find("self") != std::string::npos;
            }))
            return std::vector<ov::frontend::Place::Ptr>(m_inputs.begin() + 1, m_inputs.end());
    }
    return m_inputs;
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    return m_outputs;
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    if (tensor_name.empty())
        return {};
    auto place_it = m_name_to_place.find(tensor_name);
    if (place_it != m_name_to_place.end()) {
        return place_it->second;
    } else {
        // Return fake place that can be used to change shape or type of inputs that will exist after conversion
        auto place = std::make_shared<pytorch::Place>(*this, tensor_name, 0);
        return place;
    }
}

Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    // Return place that can be used to change shape or type of inputs that will exist after conversion
    auto place = std::make_shared<pytorch::Place>(*this, "", input_idx);
    return place;
}

void InputModel::set_partial_shape(const Place::Ptr& place, const ov::PartialShape& shape) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting shape.");
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
    if (pytorch_place->m_is_fake) {
        bool is_new = true;
        for (auto& p : m_requested_places) {
            if (p->is_equal(pytorch_place)) {
                is_new = false;
                pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(p);
                FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
            }
        }
        if (is_new)
            m_requested_places.push_back(place);
    }
    pytorch_place->m_pshape = shape;
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(
        pytorch_place,
        "Provided place is invalid. Only place of input or output is supported by PyTorch Frontend.");
    return pytorch_place->get_partial_shape();
}

void InputModel::set_element_type(const Place::Ptr& place, const ov::element::Type& type) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting element type.");
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
    if (pytorch_place->m_is_fake) {
        bool is_new = true;
        for (auto& p : m_requested_places) {
            if (p->is_equal(pytorch_place)) {
                is_new = false;
                pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(p);
                FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
            }
        }
        if (is_new)
            m_requested_places.push_back(place);
    }
    pytorch_place->m_type = type;
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(
        pytorch_place,
        "Provided place is invalid. Only place of input or output is supported by PyTorch Frontend.");
    return pytorch_place->get_element_type();
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting tensor value.");
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
    const auto& el_type = pytorch_place->m_type;
    const auto& p_shape = pytorch_place->m_pshape;
    FRONT_END_GENERAL_CHECK(el_type.is_static() && p_shape.is_static(),
                            "Shape and type must be statically defined before calling set_tensor_value");
    auto const_node = ov::op::v0::Constant::create(el_type, p_shape.to_shape(), value);
    const auto tensor_id = pytorch_place->get_tensor_index();
    auto it = m_descriptors.find(tensor_id);
    if (it != m_descriptors.end()) {
        it->second.m_value = const_node;
    } else {
        m_descriptors.emplace(tensor_id, PlaceDesc(const_node));
    }
    // remove place from inputs
    m_inputs.erase(std::remove_if(m_inputs.begin(),
                                  m_inputs.end(),
                                  [&](const Place::Ptr& p) {
                                      return p->is_equal(place);
                                  }),
                   m_inputs.end());
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
    // We need to add back "self" input if it was in initial inputs
    if (!m_inputs.empty() && m_inputs[0]) {
        // We need to remove "self" input to not confuse external users
        const auto& names = m_inputs[0]->get_names();
        if (std::any_of(names.cbegin(), names.cend(), [](const std::string& n) {
                return n.find("self") != std::string::npos;
            })) {
            FRONT_END_GENERAL_CHECK(inputs.size() == m_inputs.size() - 1,
                                    "Number of inputs provided is incorrect. Graph modification is not supported for "
                                    "this model. Expected number of inputs: ",
                                    m_inputs.size() - 1,
                                    " received ",
                                    inputs.size());
            auto self_place = m_inputs[0];
            // Verify that no same place already in vector
            auto no_self = std::none_of(inputs.cbegin(), inputs.cend(), [&](const Place::Ptr& p) {
                return p->is_equal(self_place);
            });
            FRONT_END_GENERAL_CHECK(no_self, "Unexpected input of 'self' was provided to override_all_inputs.");
            m_inputs.clear();
            m_inputs.reserve(inputs.size() + 1);
            m_inputs.push_back(std::move(self_place));
            m_inputs.insert(m_inputs.end(), inputs.cbegin(), inputs.cend());
            return;
        }
    }
    FRONT_END_GENERAL_CHECK(inputs.size() == m_inputs.size(),
                            "Number of inputs provided is incorrect. Graph modification is not supported for "
                            "this model. Expected number of inputs: ",
                            m_inputs.size(),
                            " received ",
                            inputs.size());
    m_inputs = inputs;
}

const std::string& InputModel::decoder_type_name() const {
    return m_decoder_type_name;
}

std::shared_ptr<TorchDecoder> InputModel::get_decoder() const {
    return m_model_decoder;
}

void InputModel::flush_places() {
    auto input_places = get_inputs();
    if (m_requested_places.size() > input_places.size())
        return;
    for (auto place : m_requested_places) {
        auto pt_place = std::dynamic_pointer_cast<pytorch::Place>(place);
        if (!pt_place || pt_place->get_input_index() >= input_places.size() || pt_place->m_names.size() != 0)
            return;
        auto to_update_place = std::dynamic_pointer_cast<pytorch::Place>(input_places[pt_place->get_input_index()]);
        if (!to_update_place || !to_update_place->m_type.is_dynamic() || !to_update_place->m_pshape.rank().is_dynamic())
            return;
        if (pt_place->m_type.is_static())
            to_update_place->m_type = pt_place->m_type;
        if (pt_place->m_pshape.rank().is_static())
            to_update_place->m_pshape = pt_place->m_pshape;
    }
    m_requested_places = {};
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
