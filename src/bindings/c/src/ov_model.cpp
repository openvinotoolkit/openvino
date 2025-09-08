// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_model.h"

#include "common.h"

ov_status_e ov_model_const_input(const ov_model_t* model, ov_output_const_port_t** input_port) {
    if (!model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input();
        std::unique_ptr<ov_output_const_port_t> _input_port(new ov_output_const_port_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_input_by_name(const ov_model_t* model,
                                         const char* tensor_name,
                                         ov_output_const_port_t** input_port) {
    if (!model || !tensor_name || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(tensor_name);
        std::unique_ptr<ov_output_const_port_t> _input_port(new ov_output_const_port_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_input_by_index(const ov_model_t* model,
                                          const size_t index,
                                          ov_output_const_port_t** input_port) {
    if (!model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(index);
        std::unique_ptr<ov_output_const_port_t> _input_port(new ov_output_const_port_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_input(const ov_model_t* model, ov_output_port_t** input_port) {
    if (!model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->input();
        std::unique_ptr<ov_output_port_t> _input_port(new ov_output_port_t);
        _input_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_port_t** input_port) {
    if (!model || !tensor_name || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->input(tensor_name);
        std::unique_ptr<ov_output_port_t> _input_port(new ov_output_port_t);
        _input_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_port_t** input_port) {
    if (!model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->input(index);
        std::unique_ptr<ov_output_port_t> _input_port(new ov_output_port_t);
        _input_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_output(const ov_model_t* model, ov_output_const_port_t** output_port) {
    if (!model || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->output();
        std::unique_ptr<ov_output_const_port_t> _output_port(new ov_output_const_port_t);
        _output_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_output_by_index(const ov_model_t* model,
                                           const size_t index,
                                           ov_output_const_port_t** output_port) {
    if (!model || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->output(index);
        std::unique_ptr<ov_output_const_port_t> _output_port(new ov_output_const_port_t);
        _output_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_output_by_name(const ov_model_t* model,
                                          const char* tensor_name,
                                          ov_output_const_port_t** output_port) {
    if (!model || !tensor_name || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->output(tensor_name);
        std::unique_ptr<ov_output_const_port_t> _output_port(new ov_output_const_port_t);
        _output_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_output(const ov_model_t* model, ov_output_port_t** output_port) {
    if (!model || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->output();
        std::unique_ptr<ov_output_port_t> _output_port(new ov_output_port_t);
        _output_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_output_by_index(const ov_model_t* model, const size_t index, ov_output_port_t** output_port) {
    if (!model || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->output(index);
        std::unique_ptr<ov_output_port_t> _output_port(new ov_output_port_t);
        _output_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_output_by_name(const ov_model_t* model, const char* tensor_name, ov_output_port_t** output_port) {
    if (!model || !tensor_name || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->output(tensor_name);
        std::unique_ptr<ov_output_port_t> _output_port(new ov_output_port_t);
        _output_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_inputs_size(const ov_model_t* model, size_t* input_size) {
    if (!model || !input_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto input_ports = model->object->inputs();
        *input_size = input_ports.size();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_outputs_size(const ov_model_t* model, size_t* output_size) {
    if (!model || !output_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto output_ports = model->object->outputs();
        *output_size = output_ports.size();
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

inline ov_status_e partial_shape_convert_to_cpp_object(const ov_partial_shape_t* partial_shape,
                                                       std::vector<ov::Dimension>& dims) {
    if (!partial_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        if (ov_rank_is_dynamic(partial_shape->rank)) {
            // Dynamic rank
            dims.emplace_back(ov::Dimension());
        } else {
            // Static rank
            auto rank = partial_shape->rank.max;
            for (auto i = 0; i < rank; i++) {
                auto& _dim = partial_shape->dims[i];
                dims.emplace_back(_dim.min, _dim.max);
            }
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_input_by_name(const ov_model_t* model,
                                           const char* tensor_name,
                                           const ov_partial_shape_t partial_shape) {
    if (!model || !tensor_name) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<std::string, ov::PartialShape> in_shape;
        std::vector<ov::Dimension> dims;
        auto ret = partial_shape_convert_to_cpp_object(&partial_shape, dims);
        if (ret == ov_status_e::OK) {
            in_shape[tensor_name] = dims;
        } else {
            return ret;
        }
        model->object->reshape(in_shape);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape(const ov_model_t* model,
                             const char** tensor_names,
                             const ov_partial_shape_t* partial_shapes,
                             size_t size) {
    if (!model || !tensor_names || !partial_shapes || size < 1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<std::string, ov::PartialShape> in_shapes;
        std::vector<ov::Dimension> dims;
        ov_status_e ret = ov_status_e::OK;
        for (size_t i = 0; i < size; i++) {
            auto name = tensor_names[i];
            auto pshape = &partial_shapes[i];
            dims.clear();
            ret = partial_shape_convert_to_cpp_object(pshape, dims);
            if (ret == ov_status_e::OK) {
                in_shapes[name] = dims;
            } else {
                return ret;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_by_port_indexes(const ov_model_t* model,
                                             const size_t* port_indexes,
                                             const ov_partial_shape_t* partial_shapes,
                                             size_t size) {
    if (!model || !port_indexes || !partial_shapes || size < 1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<size_t, ov::PartialShape> in_shapes;
        std::vector<ov::Dimension> dims;
        ov_status_e ret = ov_status_e::OK;
        for (size_t i = 0; i < size; i++) {
            auto port_id = port_indexes[i];
            auto pshape = &partial_shapes[i];
            dims.clear();
            ret = partial_shape_convert_to_cpp_object(pshape, dims);
            if (ret == ov_status_e::OK) {
                in_shapes[port_id] = dims;
            } else {
                return ret;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_single_input(const ov_model_t* model, const ov_partial_shape_t partial_shape) {
    size_t port = 0;
    return ov_model_reshape_by_port_indexes(model, &port, &partial_shape, 1);
}

ov_status_e ov_model_reshape_by_ports(const ov_model_t* model,
                                      const ov_output_port_t** output_ports,
                                      const ov_partial_shape_t* partial_shapes,
                                      size_t size) {
    if (!model || !output_ports || !partial_shapes || size < 1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<ov::Output<ov::Node>, ov::PartialShape> in_shapes;
        std::vector<ov::Dimension> dims;
        ov_status_e ret = ov_status_e::OK;
        for (size_t i = 0; i < size; i++) {
            auto node = *output_ports[i]->object;
            auto pshape = &partial_shapes[i];
            dims.clear();
            ret = partial_shape_convert_to_cpp_object(pshape, dims);
            if (ret == ov_status_e::OK) {
                in_shapes[node] = dims;
            } else {
                return ret;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name) {
    if (!model || !friendly_name) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto& result = model->object->get_friendly_name();
        *friendly_name = str_to_char_array(result);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_model_free(ov_model_t* model) {
    if (model)
        delete model;
}
