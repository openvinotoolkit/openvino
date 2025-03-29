// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include "input_model.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/util/log.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

Place::Place(const ov::frontend::InputModel& input_model, size_t tensor_index)
    : m_input_model(input_model),
      m_tensor_index(tensor_index) {
    const auto im = dynamic_cast<const ov::frontend::pytorch::InputModel*>(&m_input_model);
    FRONT_END_GENERAL_CHECK(im, "PyTorch Place requires PyTorch InputModel class.");
    auto decoder = im->get_decoder();
    const auto& inputs = decoder->inputs();
    const auto& outputs = decoder->outputs();
    auto in_it = std::find(inputs.begin(), inputs.end(), tensor_index);
    if (in_it != inputs.end()) {
        m_is_input = true;
        m_input_index = std::distance(inputs.begin(), in_it);
        auto idx = std::distance(inputs.begin(), in_it);
        const auto& signature_name = decoder->get_input_signature_name(idx);
        m_names.push_back(signature_name);

        auto type_any = simplified_type_interpret(decoder->get_input_type(idx));
        if (type_any.is<element::Type>()) {
            m_type = type_any.as<element::Type>();
        }
        m_pshape = decoder->get_input_shape(idx);
    }
    auto out_it = std::find(outputs.begin(), outputs.end(), tensor_index);
    if (out_it != outputs.end()) {
        m_is_output = true;
        if (!m_is_input) {
            auto idx = std::distance(outputs.begin(), out_it);
            const auto& debug_name = decoder->get_output_debug_name(idx);
            m_names.push_back(debug_name);

            auto type_any = simplified_type_interpret(decoder->get_output_type(idx));
            if (type_any.is<element::Type>()) {
                m_type = type_any.as<element::Type>();
            }
            m_pshape = decoder->get_output_shape(idx);
        }
    }
    if (m_is_input && m_is_output) {
        OPENVINO_DEBUG("[WARNING] Place ", tensor_index, " is input and output at a same time.");
    }
}

Place::Place(const ov::frontend::InputModel& input_model, const std::string& name, size_t input_index)
    : m_input_model(input_model),
      m_tensor_index(0),
      m_is_fake(true),
      m_input_index(input_index),
      m_pshape(PartialShape::dynamic()),
      m_type(element::dynamic),
      m_is_input(true) {
    if (!name.empty())
        m_names = {name};
}

bool Place::is_equal(const Ptr& another) const {
    const auto& pt_place = std::dynamic_pointer_cast<pytorch::Place>(another);
    if (!pt_place)
        return this == another.get();
    if (m_is_fake || pt_place->m_is_fake) {
        if ((m_is_fake && m_names.size() != 0) || (pt_place->m_is_fake && pt_place->m_names.size() != 0))
            // named fake place can only be equal to itself
            return this == another.get();
        return m_input_index == pt_place->get_input_index();
    }
    return this == another.get();
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
