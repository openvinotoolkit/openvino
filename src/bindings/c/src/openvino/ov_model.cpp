// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/ov_model.h"

#include "common.h"

ov_status_e ov_model_outputs(const ov_model_t* model, ov_output_node_list_t* output_nodes) {
    if (!model || !output_nodes) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto results = std::const_pointer_cast<const ov::Model>(model->object)->outputs();
        output_nodes->num = results.size();
        auto tmp_output_nodes(new ov_output_const_node_t[output_nodes->num]);

        for (size_t i = 0; i < output_nodes->num; i++) {
            tmp_output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(results[i]));
        }
        output_nodes->output_nodes = tmp_output_nodes;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_inputs(const ov_model_t* model, ov_output_node_list_t* input_nodes) {
    if (!model || !input_nodes) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto results = std::const_pointer_cast<const ov::Model>(model->object)->inputs();
        input_nodes->num = results.size();
        auto tmp_output_nodes(new ov_output_const_node_t[input_nodes->num]);

        for (size_t i = 0; i < input_nodes->num; i++) {
            tmp_output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(results[i]));
        }
        input_nodes->output_nodes = tmp_output_nodes;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_node_get_any_name(ov_output_const_node_t* node, char** tensor_name) {
    if (!node || !tensor_name) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        *tensor_name = str_to_char_array(node->object->get_any_name());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_any_name_by_index(ov_output_node_list_t* nodes, size_t idx, char** tensor_name) {
    if (!nodes || !tensor_name || idx >= nodes->num) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        *tensor_name = str_to_char_array(nodes->output_nodes[idx].object->get_any_name());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_shape(ov_output_const_node_t* node, ov_shape_t* tensor_shape) {
    if (!node || !tensor_shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto shape = node->object->get_shape();
        if (shape.size() > MAX_DIMENSION) {
            return ov_status_e::INVALID_PARAM;
        }
        tensor_shape->rank = shape.size();
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_shape_by_index(ov_output_node_list_t* nodes, size_t idx, ov_shape_t* tensor_shape) {
    if (!nodes || idx >= nodes->num || !tensor_shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto shape = nodes->output_nodes[idx].object->get_shape();
        if (shape.size() > MAX_DIMENSION) {
            return ov_status_e::GENERAL_ERROR;
        }
        tensor_shape->rank = shape.size();
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_partial_shape_by_index(ov_output_node_list_t* nodes,
                                               size_t idx,
                                               ov_partial_shape_t** partial_shape) {
    if (!nodes || idx >= nodes->num || !partial_shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_partial_shape_t> _partial_shape(new ov_partial_shape_t);
        auto shape = nodes->output_nodes[idx].object->get_partial_shape();

        _partial_shape->rank = shape.rank();
        auto iter = shape.begin();
        for (; iter != shape.end(); iter++)
            _partial_shape->dims.emplace_back(*iter);
        *partial_shape = _partial_shape.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_element_type_by_index(ov_output_node_list_t* nodes,
                                              size_t idx,
                                              ov_element_type_e* tensor_type) {
    if (!nodes || idx >= nodes->num) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto type = (ov::element::Type_t)nodes->output_nodes[idx].object->get_element_type();
        *tensor_type = (ov_element_type_e)type;
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_element_type(ov_output_const_node_t* node, ov_element_type_e* tensor_type) {
    if (!node || !tensor_type) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto type = (ov::element::Type_t)node->object->get_element_type();
        *tensor_type = (ov_element_type_e)type;
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_model_input_by_name(const ov_model_t* model,
                                   const char* tensor_name,
                                   ov_output_const_node_t** input_node) {
    if (!model || !tensor_name || !input_node) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(tensor_name);
        *input_node = new ov_output_const_node_t;
        (*input_node)->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_const_node_t** input_node) {
    if (!model || !input_node) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(index);
        *input_node = new ov_output_const_node_t;
        (*input_node)->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

bool ov_model_is_dynamic(const ov_model_t* model) {
    if (!model) {
        printf("[ERROR] The model is NULL!!!\n");
        return false;
    }
    return model->object->is_dynamic();
}

template <class T>
T str_to_value(const std::string& str) {
    T ret{0};
    std::istringstream ss(str);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}

ov_status_e ov_model_reshape_by_name(const ov_model_t* model,
                                     const char* tensor_name,
                                     const ov_partial_shape_t* partial_shape) {
    if (!model || !tensor_name || !partial_shape) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::map<std::string, ov::PartialShape> in_shape;
        if (partial_shape->rank.is_static() && (partial_shape->rank.get_length() == partial_shape->dims.size())) {
            in_shape[tensor_name] = partial_shape->dims;
        } else {
            return ov_status_e::PARAMETER_MISMATCH;
        }
        model->object->reshape(in_shape);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_by_names(const ov_model_t* model,
                                      const char* tensor_names[],
                                      const ov_partial_shape_t* partial_shapes[],
                                      size_t cnt) {
    if (!model || !tensor_names || !partial_shapes || cnt < 1) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::map<std::string, ov::PartialShape> in_shapes;
        for (auto i = 0; i < cnt; i++) {
            auto name = tensor_names[i];
            if (partial_shapes[i]->rank.is_static() &&
                (partial_shapes[i]->rank.get_length() == partial_shapes[i]->dims.size())) {
                in_shapes[name] = partial_shapes[i]->dims;
            } else {
                return ov_status_e::PARAMETER_MISMATCH;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_by_ports(const ov_model_t* model,
                                      size_t* ports,
                                      const ov_partial_shape_t** partial_shape,
                                      size_t cnt) {
    if (!model || !ports || !partial_shape || cnt < 1) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::map<size_t, ov::PartialShape> in_shapes;
        for (auto i = 0; i < cnt; i++) {
            auto port_id = ports[i];
            if (partial_shape[i]->rank.is_static() &&
                (partial_shape[i]->rank.get_length() == partial_shape[i]->dims.size())) {
                in_shapes[port_id] = partial_shape[i]->dims;
            } else {
                return ov_status_e::PARAMETER_MISMATCH;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape(const ov_model_t* model, const ov_partial_shape_t* partial_shape) {
    size_t port = 0;
    return ov_model_reshape_by_ports(model, &port, &partial_shape, 1);
}

ov_status_e ov_model_reshape_by_nodes(const ov_model_t* model,
                                      const ov_output_node_t* output_nodes[],
                                      const ov_partial_shape_t* partial_shapes[],
                                      size_t cnt) {
    if (!model || !output_nodes || !partial_shapes || cnt < 1) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::map<ov::Output<ov::Node>, ov::PartialShape> in_shapes;
        for (auto i = 0; i < cnt; i++) {
            auto node = *output_nodes[i]->object;
            if (partial_shapes[i]->rank.is_static() &&
                (partial_shapes[i]->rank.get_length() == partial_shapes[i]->dims.size())) {
                in_shapes[node] = partial_shapes[i]->dims;
            } else {
                return ov_status_e::PARAMETER_MISMATCH;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name) {
    if (!model || !friendly_name) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto& result = model->object->get_friendly_name();
        *friendly_name = str_to_char_array(result);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_output_node_list_free(ov_output_node_list_t* output_nodes) {
    if (output_nodes) {
        if (output_nodes->output_nodes)
            delete[] output_nodes->output_nodes;
        output_nodes->output_nodes = nullptr;
    }
}

void ov_output_node_free(ov_output_const_node_t* output_node) {
    if (output_node)
        delete output_node;
}

void ov_model_free(ov_model_t* model) {
    if (model)
        delete model;
}

void ov_free(const char* content) {
    if (content)
        delete content;
}
