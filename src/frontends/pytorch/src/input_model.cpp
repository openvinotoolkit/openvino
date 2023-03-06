// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

InputModel::InputModel(std::shared_ptr<TorchDecoder> model_decoder) : m_model_decoder(model_decoder) {
    const auto& inputs = m_model_decoder->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto in_place = std::make_shared<pytorch::Place>(*this, inputs[i]);
        for (const auto& name : in_place->get_names()) {
            m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(in_place));
        }
        auto type_any = simplified_type_interpret(m_model_decoder->get_input_type(i));
        auto dtype = element::dynamic;
        if (type_any.is<element::Type>()) {
            dtype = type_any.as<element::Type>();
        }
        m_descriptors.emplace(inputs[i], PlaceDesc(dtype, m_model_decoder->get_input_shape(i)));
    }
    const auto& outputs = m_model_decoder->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto out_place = std::make_shared<pytorch::Place>(*this, outputs[i]);
        for (const auto& name : out_place->get_names()) {
            m_name_to_place.emplace(name, std::dynamic_pointer_cast<frontend::Place>(out_place));
        }
        auto type_any = simplified_type_interpret(m_model_decoder->get_output_type(i));
        auto dtype = element::dynamic;
        if (type_any.is<element::Type>()) {
            dtype = type_any.as<element::Type>();
        }
        m_descriptors.emplace(outputs[i], PlaceDesc(dtype, m_model_decoder->get_output_shape(i)));
    }
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_inputs() const {
    std::vector<ov::frontend::Place::Ptr> res;
    for (const auto& input_idx : m_model_decoder->inputs()) {
        auto place_it = m_name_to_place.find(std::to_string(input_idx));
        FRONT_END_GENERAL_CHECK(place_it != m_name_to_place.end(), "Couldn't find Place for input.");
        res.push_back(place_it->second);
    }
    return res;
}

std::vector<ov::frontend::Place::Ptr> InputModel::get_outputs() const {
    std::vector<ov::frontend::Place::Ptr> res;
    for (const auto& output_idx : m_model_decoder->outputs()) {
        auto place_it = m_name_to_place.find(std::to_string(output_idx));
        FRONT_END_GENERAL_CHECK(place_it != m_name_to_place.end(), "Couldn't find Place for output.");
        res.push_back(place_it->second);
    }
    return res;
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
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
    auto it = m_descriptors.find(pytorch_place->get_tensor_index());
    if (it != m_descriptors.end()) {
        it->second.m_pshape = shape;
    }
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(
        pytorch_place,
        "Provided place is invalid. Only place of input or output is supported by PyTorch Frontend.");
    auto it = m_descriptors.find(pytorch_place->get_tensor_index());
    if (it != m_descriptors.end()) {
        return it->second.m_pshape;
    }
    return PartialShape::dynamic();
}

void InputModel::set_element_type(const Place::Ptr& place, const ov::element::Type& type) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting element type.");
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
    auto it = m_descriptors.find(pytorch_place->get_tensor_index());
    if (it != m_descriptors.end()) {
        it->second.m_type = type;
    }
}

ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(
        pytorch_place,
        "Provided place is invalid. Only place of input or output is supported by PyTorch Frontend.");
    auto it = m_descriptors.find(pytorch_place->get_tensor_index());
    if (it != m_descriptors.end()) {
        return it->second.m_type;
    }
    return element::dynamic;
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    FRONT_END_GENERAL_CHECK(place && place->is_input(),
                            "Provided place is invalid, only inputs are supported for setting tensor value.");
    auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(place);
    FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported");
    auto it = m_descriptors.find(pytorch_place->get_tensor_index());
    if (it != m_descriptors.end()) {
        auto el_type = it->second.m_type;
        auto p_shape = it->second.m_pshape;
        FRONT_END_GENERAL_CHECK(el_type.is_static() && p_shape.is_static(),
                                "Shape and type must be statically defined before calling set_tensor_value");
        it->second.m_value = ov::op::v0::Constant::create(el_type, p_shape.to_shape(), value);
    } else {
        FRONT_END_GENERAL_CHECK(false, "Place is not known.");
    }
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
