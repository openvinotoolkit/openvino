// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <graph_iterator_flatbuffer.hpp>

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderFlatBuffer>
ov::frontend::tensorflow_lite::GraphIteratorFlatBuffer::get_decoder() const {
    auto inputs_vec = (*m_model->subgraphs()->begin())->inputs();
    auto outputs_vec = (*m_model->subgraphs()->begin())->outputs();
    auto inputs = std::set<int32_t>{inputs_vec->begin(), inputs_vec->end()};
    auto outputs = std::set<int32_t>{outputs_vec->begin(), outputs_vec->end()};

    auto buffers = m_model->buffers();
    auto tensors = m_model->subgraphs()->begin()->tensors();

    std::map<size_t, TensorInfo> input_info = {}, output_info = {};
    size_t i = 0;
    for (auto input : *m_nodes[node_index]->inputs()) {
        if (input == -1) {
            continue;
        }
        auto buffer = (*buffers)[(*tensors)[input]->buffer()];
        auto is_input = inputs.find(input) != inputs.end();
        int64_t input_idx =
            !is_input ? -1 : std::find(inputs_vec->begin(), inputs_vec->end(), input) - inputs_vec->begin();
        auto is_output = outputs.find(input) != outputs.end();
        int64_t output_idx =
            !is_output ? -1 : std::find(outputs_vec->begin(), outputs_vec->end(), input) - outputs_vec->begin();
        input_info[i++] = TensorInfo{input_idx, output_idx, (*tensors)[input], buffer};
    }
    i = 0;
    // If we have any m_nodes[node_index]->intermediates() than trigger internal smth? no
    // put all the info in Decoder as a sub-graph!

    for (auto output : *m_nodes[node_index]->outputs()) {
        auto buffer = (*buffers)[(*tensors)[output]->buffer()];
        auto is_output = outputs.find(output) != outputs.end();
        int64_t output_idx =
            !is_output ? -1 : std::find(outputs_vec->begin(), outputs_vec->end(), output) - outputs_vec->begin();
        output_info[i++] = TensorInfo{-1, output_idx, (*tensors)[output], buffer};
    }
    auto op_codes = m_model->operator_codes();
    auto operator_code = (*op_codes)[m_nodes[node_index]->opcode_index()];
    std::string type;
    if (operator_code->deprecated_builtin_code() <
        tflite::BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES) {
        type = tflite::EnumNamesBuiltinOperator()[operator_code->deprecated_builtin_code()];
    } else {
        type = tflite::EnumNamesBuiltinOperator()[operator_code->builtin_code()];
    }
    return std::make_shared<ov::frontend::tensorflow_lite::DecoderFlatBuffer>(m_nodes[node_index],
                                                                              type,
                                                                              std::to_string(node_index),
                                                                              input_info,
                                                                              output_info);
}
