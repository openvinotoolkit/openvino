// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include "input_model.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

Place::Place(const ov::frontend::InputModel& input_model, size_t tensor_index)
    : m_input_model(input_model),
      m_tensor_index(tensor_index),
      m_is_input(false),
      m_is_output(false) {
    m_names.push_back(std::to_string(tensor_index));
    const auto im = dynamic_cast<const ov::frontend::pytorch::InputModel*>(&m_input_model);
    FRONT_END_GENERAL_CHECK(im, "PyTorch Place requires PyTorch InputModel class.");
    const auto& inputs = im->m_model_decoder->inputs();
    const auto& outputs = im->m_model_decoder->outputs();
    auto in_it = std::find(inputs.begin(), inputs.end(), tensor_index);
    if (in_it != inputs.end()) {
        m_is_input = true;
        const auto& debug_name = im->m_model_decoder->get_input_debug_name(std::distance(inputs.begin(), in_it));
        if (debug_name != m_names.at(0)) {
            m_names.push_back(debug_name);
        }
    }
    auto out_it = std::find(outputs.begin(), outputs.end(), tensor_index);
    if (out_it != outputs.end()) {
        m_is_output = true;
        const auto& debug_name = im->m_model_decoder->get_output_debug_name(std::distance(outputs.begin(), out_it));
        if (debug_name != m_names.at(0)) {
            m_names.push_back(debug_name);
        }
    }
    if (m_is_input && m_is_output) {
        OPENVINO_DEBUG << "[WARNING] Place " << tensor_index << " is input and output at a same time.";
    }
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
