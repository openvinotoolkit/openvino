// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_compiled_model.h"

#include <stdarg.h>

#include "common.h"

//!<  Read-only property<char *> to get a string list of supported read-only properties.
const char* ov_property_key_supported_properties_ = "SUPPORTED_PROPERTIES";

ov_status_e ov_compiled_model_inputs_size(const ov_compiled_model_t* compiled_model, size_t* input_size) {
    if (!compiled_model || !input_size) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto input_ports = compiled_model->object->inputs();
        *input_size = input_ports.size();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_input(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** input_port) {
    if (!compiled_model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto result = std::const_pointer_cast<const ov::CompiledModel>(compiled_model->object)->input();
        std::unique_ptr<ov_output_const_port_t> _input_port(new ov_output_const_port_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_input_by_index(const ov_compiled_model_t* compiled_model,
                                             const size_t index,
                                             ov_output_const_port_t** input_port) {
    if (!compiled_model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto result = std::const_pointer_cast<const ov::CompiledModel>(compiled_model->object)->input(index);
        std::unique_ptr<ov_output_const_port_t> _input_port(new ov_output_const_port_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_input_by_name(const ov_compiled_model_t* compiled_model,
                                            const char* name,
                                            ov_output_const_port_t** input_port) {
    if (!compiled_model || !name || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto result = std::const_pointer_cast<const ov::CompiledModel>(compiled_model->object)->input(name);
        std::unique_ptr<ov_output_const_port_t> _input_port(new ov_output_const_port_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_outputs_size(const ov_compiled_model_t* compiled_model, size_t* output_size) {
    if (!compiled_model || !output_size) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto output_ports = compiled_model->object->outputs();
        *output_size = output_ports.size();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_output(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** output_port) {
    if (!compiled_model || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto result = std::const_pointer_cast<const ov::CompiledModel>(compiled_model->object)->output();
        std::unique_ptr<ov_output_const_port_t> _output_port(new ov_output_const_port_t);
        _output_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_output_by_index(const ov_compiled_model_t* compiled_model,
                                              const size_t index,
                                              ov_output_const_port_t** output_port) {
    if (!compiled_model || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto result = std::const_pointer_cast<const ov::CompiledModel>(compiled_model->object)->output(index);
        std::unique_ptr<ov_output_const_port_t> _output_port(new ov_output_const_port_t);
        _output_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_output_by_name(const ov_compiled_model_t* compiled_model,
                                             const char* name,
                                             ov_output_const_port_t** output_port) {
    if (!compiled_model || !name || !output_port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto result = std::const_pointer_cast<const ov::CompiledModel>(compiled_model->object)->output(name);
        std::unique_ptr<ov_output_const_port_t> _output_port(new ov_output_const_port_t);
        _output_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_port = _output_port.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model, ov_model_t** model) {
    if (!compiled_model || !model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        auto runtime_model = compiled_model->object->get_runtime_model();
        _model->object = std::const_pointer_cast<ov::Model>(std::move(runtime_model));
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model,
                                                   ov_infer_request_t** infer_request) {
    if (!compiled_model || !infer_request) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_infer_request_t> _infer_request(new ov_infer_request_t);
        auto infer_req = compiled_model->object->create_infer_request();
        _infer_request->object = std::make_shared<ov::InferRequest>(std::move(infer_req));
        *infer_request = _infer_request.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, ...) {
    if (!compiled_model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap property = {};

        va_list args_ptr;
        va_start(args_ptr, compiled_model);
        GET_PROPERTY_FROM_ARGS_LIST;
        va_end(args_ptr);

        compiled_model->object->set_property(property);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                                           const char* key,
                                           char** property_value) {
    if (!compiled_model || !key || !property_value) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto value = compiled_model->object->get_property(key);
        *property_value = str_to_char_array(value.as<std::string>());
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_export_model(const ov_compiled_model_t* compiled_model, const char* export_model_path) {
    if (!compiled_model || !export_model_path) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::ofstream model_file(export_model_path, std::ios::out | std::ios::binary);
        if (model_file.is_open()) {
            compiled_model->object->export_model(model_file);
        } else {
            return ov_status_e::GENERAL_ERROR;
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_compiled_model_free(ov_compiled_model_t* compiled_model) {
    if (compiled_model)
        delete compiled_model;
}

ov_status_e ov_compiled_model_get_context(const ov_compiled_model_t* compiled_model, ov_remote_context** context) {
    if (!compiled_model || !context) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::RemoteContext object = compiled_model->object->get_context();
        std::unique_ptr<ov_remote_context> _context(new ov_remote_context);
        _context->object = std::make_shared<ov::RemoteContext>(std::move(object));
        *context = _context.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}
