// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_compiled_model.h"

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

inline ov_status_e ov_compiled_model_properies_to_anymap(const ov_properties_t* properties, ov::AnyMap& dest) {
    if (!properties || properties->size <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }

    return ov_status_e::NOT_IMPLEMENTED;
}

ov_status_e ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, const ov_properties_t* property) {
    if (!compiled_model || !property) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap dest;
        auto ret = ov_compiled_model_properies_to_anymap(property, dest);
        if (ret == ov_status_e::OK) {
            compiled_model->object->set_property(dest);
        } else {
            return ret;
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                                           const char* key,
                                           ov_any_t* value) {
    if (!compiled_model || !value || !key) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::string _key = std::string(key);
        if (_key == ov_property_key_supported_properties_) {
            auto supported_properties = compiled_model->object->get_property(ov::supported_properties);
            std::string tmp_s;
            for (const auto& i : supported_properties) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* temp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.c_str(), tmp_s.length() + 1, temp);
            value->ptr = static_cast<void*>(temp);
            value->size = tmp_s.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else {
            return ov_status_e::NOT_IMPLEMENTED;
        }
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
