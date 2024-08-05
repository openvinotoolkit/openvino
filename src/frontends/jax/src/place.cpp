// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include "input_model.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/util/log.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {

Place::Place(const ov::frontend::InputModel& input_model, size_t tensor_index)
    : m_input_model(input_model),
      m_tensor_index(tensor_index) {
    const auto im = dynamic_cast<const ov::frontend::jax::InputModel*>(&m_input_model);
    FRONT_END_GENERAL_CHECK(im, "Jax Place requires Jax InputModel class.");
    auto decoder = im->get_decoder();
    const auto& inputs = decoder->inputs();
    const auto& outputs = decoder->outputs();
    auto in_it = std::find(inputs.begin(), inputs.end(), tensor_index);
    if (in_it != inputs.end()) {
        m_is_input = true;
        auto idx = std::distance(inputs.begin(), in_it);
        const auto& signature_name = decoder->get_input_signature_name(idx);
        m_names.push_back(signature_name);

        auto type_any = decoder->get_input_type(idx);
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
            const auto& debug_name = decoder->get_output_name(idx);
            m_names.push_back(debug_name);

            auto type_any = decoder->get_output_type(idx);
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

}  // namespace jax
}  // namespace frontend
}  // namespace ov
