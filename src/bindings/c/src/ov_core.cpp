// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_core.h"

#include <stdarg.h>

#include "common.h"

char* str_to_char_array(const std::string& str) {
    std::unique_ptr<char> _char_array(new char[str.length() + 1]);
    char* char_array = _char_array.release();
    std::copy_n(str.begin(), str.length() + 1, char_array);
    return char_array;
}

ov_status_e ov_get_openvino_version(ov_version_t* version) {
    if (!version) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::Version object = ov::get_openvino_version();

        std::string version_builderNumber = object.buildNumber;
        version->buildNumber = str_to_char_array(version_builderNumber);

        std::string version_description = object.description;
        version->description = str_to_char_array(version_description);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_version_free(ov_version_t* version) {
    if (!version) {
        return;
    }
    delete[] version->buildNumber;
    version->buildNumber = nullptr;
    delete[] version->description;
    version->description = nullptr;
}

ov_status_e ov_core_create_with_config(const char* xml_config_file, ov_core_t** core) {
    if (!core || !xml_config_file) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_core_t> _core(new ov_core_t);
        _core->object = std::make_shared<ov::Core>(xml_config_file);
        *core = _core.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_create(ov_core_t** core) {
    return ov_core_create_with_config("", core);
}

void ov_core_free(ov_core_t* core) {
    if (core)
        delete core;
}

ov_status_e ov_core_read_model(const ov_core_t* core,
                               const char* model_path,
                               const char* bin_path,
                               ov_model_t** model) {
    if (!core || !model_path || !model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::string bin = "";
        if (bin_path) {
            bin = bin_path;
        }
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        _model->object = core->object->read_model(model_path, bin);
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_read_model_from_memory(const ov_core_t* core,
                                           const char* model_str,
                                           const ov_tensor_t* weights,
                                           ov_model_t** model) {
    if (!core || !model_str || !model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        if (weights) {
            _model->object = core->object->read_model(model_str, *(weights->object));
        } else {
            _model->object = core->object->read_model(model_str, ov::Tensor());
        }
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_compile_model(const ov_core_t* core,
                                  const ov_model_t* model,
                                  const char* device_name,
                                  bool with_property,
                                  ov_compiled_model_t** compiled_model,
                                  ...) {
    if (!core || !model || !compiled_model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap property = {};
        if (with_property) {
            va_list args_ptr;
            va_start(args_ptr, compiled_model);
            std::vector<ov::Any> value_vec;

            // Only support one or none property key.
            // Each key can be followed by single data type or compound type
            std::string property_key = va_arg(args_ptr, char*);
            GET_ONE_PROPERTY_FROM_ARGS_LIST(1)
        }
        std::string dev_name = "";
        ov::CompiledModel object;
        if (device_name) {
            dev_name = device_name;
            object = core->object->compile_model(model->object, dev_name, property);
        } else {
            object = core->object->compile_model(model->object, property);
        }
        std::unique_ptr<ov_compiled_model_t> _compiled_model(new ov_compiled_model_t);
        _compiled_model->object = std::make_shared<ov::CompiledModel>(std::move(object));
        *compiled_model = _compiled_model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_compile_model_from_file(const ov_core_t* core,
                                            const char* model_path,
                                            const char* device_name,
                                            bool with_property,
                                            ov_compiled_model_t** compiled_model,
                                            ...) {
    if (!core || !model_path || !compiled_model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap property = {};
        if (with_property) {
            va_list args_ptr;
            va_start(args_ptr, compiled_model);
            std::vector<ov::Any> value_vec;

            // Only support one or zero property key.
            // Each key can be followed by single data type or compound type
            std::string property_key = va_arg(args_ptr, char*);
            GET_ONE_PROPERTY_FROM_ARGS_LIST(1)
        }

        ov::CompiledModel object;
        std::string dev_name = "";
        if (device_name) {
            dev_name = device_name;
            object = core->object->compile_model(model_path, dev_name, property);
        } else {
            object = core->object->compile_model(model_path, property);
        }
        std::unique_ptr<ov_compiled_model_t> _compiled_model(new ov_compiled_model_t);
        _compiled_model->object = std::make_shared<ov::CompiledModel>(std::move(object));
        *compiled_model = _compiled_model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

const std::map<ov_performance_mode_e, ov::hint::PerformanceMode> performance_mode_map = {
    {ov_performance_mode_e::UNDEFINED_MODE, ov::hint::PerformanceMode::UNDEFINED},
    {ov_performance_mode_e::THROUGHPUT, ov::hint::PerformanceMode::THROUGHPUT},
    {ov_performance_mode_e::LATENCY, ov::hint::PerformanceMode::LATENCY},
    {ov_performance_mode_e::CUMULATIVE_THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT}};

ov_status_e ov_core_set_property(const ov_core_t* core, const char* device_name, const char* property_key, ...) {
    if (!core || !property_key) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, property_key);

        std::vector<ov::Any> value_vec;
        GET_ONE_PROPERTY_FROM_ARGS_LIST(1)

        if (property.size() == 0) {
            return ov_status_e::INVALID_C_PARAM;
        }

        if (device_name) {
            core->object->set_property(device_name, property);
        } else {
            core->object->set_property(property);
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_property(const ov_core_t* core,
                                 const char* device_name,
                                 const char* property_key,
                                 char** property_value) {
    if (!core || !property_key || !property_value) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto value = core->object->get_property(device_name, property_key);
        *property_value = str_to_char_array(value.as<std::string>());
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_available_devices(const ov_core_t* core, ov_available_devices_t* devices) {
    if (!core) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto available_devices = core->object->get_available_devices();
        devices->size = available_devices.size();
        std::unique_ptr<char*[]> tmp_devices(new char*[available_devices.size()]);
        for (size_t i = 0; i < available_devices.size(); i++) {
            tmp_devices[i] = str_to_char_array(available_devices[i]);
        }
        devices->devices = tmp_devices.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_available_devices_free(ov_available_devices_t* devices) {
    if (!devices) {
        return;
    }
    for (size_t i = 0; i < devices->size; i++) {
        if (devices->devices[i]) {
            delete[] devices->devices[i];
        }
    }
    if (devices->devices)
        delete[] devices->devices;
    devices->devices = nullptr;
    devices->size = 0;
}

ov_status_e ov_core_import_model(const ov_core_t* core,
                                 const char* content,
                                 const size_t content_size,
                                 const char* device_name,
                                 ov_compiled_model_t** compiled_model) {
    if (!core || !content || !device_name || !compiled_model) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        mem_istream model_stream(content, content_size);
        std::unique_ptr<ov_compiled_model_t> _compiled_model(new ov_compiled_model_t);
        auto object = core->object->import_model(model_stream, device_name);
        _compiled_model->object = std::make_shared<ov::CompiledModel>(std::move(object));
        *compiled_model = _compiled_model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_versions_by_device_name(const ov_core_t* core,
                                                const char* device_name,
                                                ov_core_version_list_t* versions) {
    if (!core || !device_name || !versions) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto object = core->object->get_versions(device_name);
        if (object.empty()) {
            return ov_status_e::NOT_FOUND;
        }
        versions->size = object.size();
        auto tmp_versions(new ov_core_version_t[object.size()]);
        auto iter = object.cbegin();
        for (size_t i = 0; i < object.size(); i++, iter++) {
            const auto& tmp_version_name = iter->first;
            tmp_versions[i].device_name = str_to_char_array(tmp_version_name);

            const auto tmp_version_build_number = iter->second.buildNumber;
            tmp_versions[i].version.buildNumber = str_to_char_array(tmp_version_build_number);

            const auto tmp_version_description = iter->second.description;
            tmp_versions[i].version.description = str_to_char_array(tmp_version_description);
        }
        versions->versions = tmp_versions;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_core_versions_free(ov_core_version_list_t* versions) {
    if (!versions) {
        return;
    }
    for (size_t i = 0; i < versions->size; i++) {
        if (versions->versions[i].device_name)
            delete[] versions->versions[i].device_name;
        if (versions->versions[i].version.buildNumber)
            delete[] versions->versions[i].version.buildNumber;
        if (versions->versions[i].version.description)
            delete[] versions->versions[i].version.description;
    }

    if (versions->versions)
        delete[] versions->versions;
    versions->versions = nullptr;
}
