// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_core.h"

#include "common.h"

/**
 * @variable global value for error info.
 * Don't change its order.
 */
char const* error_infos[] = {"success",
                             "general error",
                             "it's not implement",
                             "failed to network",
                             "input parameter mismatch",
                             "cannot find the value",
                             "out of bounds",
                             "run with unexpected error",
                             "request is busy",
                             "result is not ready",
                             "it is not allocated",
                             "inference start with error",
                             "network is not ready",
                             "inference is canceled",
                             "invalid c input parameters",
                             "unknown c error"};

const char* ov_get_error_info(ov_status_e status) {
    auto index = -status;
    auto max_index = sizeof(error_infos) / sizeof(error_infos[0]) - 1;
    if (index > max_index)
        return error_infos[max_index];
    return error_infos[index];
}

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
                                  ov_compiled_model_t** compiled_model,
                                  const ov_property_t* property) {
    if (!core || !model || !compiled_model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::string dev_name = "";
        ov::CompiledModel object;
        if (device_name) {
            dev_name = device_name;
            object = core->object->compile_model(model->object, dev_name);
        } else {
            object = core->object->compile_model(model->object);
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
                                            ov_compiled_model_t** compiled_model,
                                            const ov_property_t* property) {
    if (!core || !model_path || !compiled_model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::CompiledModel object;
        std::string dev_name = "";
        if (device_name) {
            dev_name = device_name;
            object = core->object->compile_model(model_path, dev_name);
        } else {
            object = core->object->compile_model(model_path);
        }
        std::unique_ptr<ov_compiled_model_t> _compiled_model(new ov_compiled_model_t);
        _compiled_model->object = std::make_shared<ov::CompiledModel>(std::move(object));
        *compiled_model = _compiled_model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_set_property(const ov_core_t* core, const char* device_name, const ov_property_t* property) {
    if (!core || !property) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        if (device_name) {
            core->object->set_property(device_name, property->object);
        } else {
            core->object->set_property(property->object);
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_property(const ov_core_t* core,
                                 const char* device_name,
                                 const ov_property_key_e key,
                                 ov_property_value_t* value) {
    if (!core || !device_name || !value) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        switch (key) {
        case ov_property_key_e::SUPPORTED_PROPERTIES: {
            auto supported_properties = core->object->get_property(device_name, ov::supported_properties);
            std::string tmp_s;
            for (const auto& i : supported_properties) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* tmp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->cnt = tmp_s.length() + 1;
            value->type = ov_property_value_type_e::CHAR;
            break;
        }
        case ov_property_key_e::AVAILABLE_DEVICES: {
            auto available_devices = core->object->get_property(device_name, ov::available_devices);
            std::string tmp_s;
            for (const auto& i : available_devices) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* tmp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->cnt = tmp_s.length() + 1;
            value->type = ov_property_value_type_e::CHAR;
            break;
        }
        case ov_property_key_e::OPTIMAL_NUMBER_OF_INFER_REQUESTS: {
            auto optimal_number_of_infer_requests =
                core->object->get_property(device_name, ov::optimal_number_of_infer_requests);
            uint32_t* temp = new uint32_t;
            *temp = optimal_number_of_infer_requests;
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::UINT32;
            break;
        }
        case ov_property_key_e::RANGE_FOR_ASYNC_INFER_REQUESTS: {
            auto range = core->object->get_property(device_name, ov::range_for_async_infer_requests);
            uint32_t* temp = new uint32_t[3];
            temp[0] = std::get<0>(range);
            temp[1] = std::get<1>(range);
            temp[2] = std::get<2>(range);
            value->ptr = static_cast<void*>(temp);
            value->cnt = 3;
            value->type = ov_property_value_type_e::UINT32;
            break;
        }
        case ov_property_key_e::RANGE_FOR_STREAMS: {
            auto range = core->object->get_property(device_name, ov::range_for_streams);
            uint32_t* temp = new uint32_t[2];
            temp[0] = std::get<0>(range);
            temp[1] = std::get<1>(range);
            value->ptr = static_cast<void*>(temp);
            value->cnt = 2;
            value->type = ov_property_value_type_e::UINT32;
            break;
        }
        case ov_property_key_e::FULL_DEVICE_NAME: {
            auto name = core->object->get_property(device_name, ov::device::full_name);
            char* tmp = new char[name.length() + 1];
            std::copy_n(name.begin(), name.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->cnt = name.length() + 1;
            value->type = ov_property_value_type_e::CHAR;
            break;
        }
        case ov_property_key_e::OPTIMIZATION_CAPABILITIES: {
            auto capabilities = core->object->get_property(device_name, ov::device::capabilities);
            std::string tmp_s;
            for (const auto& i : capabilities) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* tmp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->cnt = tmp_s.length() + 1;
            value->type = ov_property_value_type_e::CHAR;
            break;
        }
        case ov_property_key_e::CACHE_DIR: {
            auto dir = core->object->get_property(device_name, ov::cache_dir);
            char* tmp = new char[dir.length() + 1];
            std::copy_n(dir.begin(), dir.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->cnt = dir.length() + 1;
            value->type = ov_property_value_type_e::CHAR;
            break;
        }
        case ov_property_key_e::NUM_STREAMS: {
            auto num = core->object->get_property(device_name, ov::num_streams);
            int32_t* temp = new int32_t;
            *temp = num.num;
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::INT32;
            break;
        }
        case ov_property_key_e::AFFINITY: {
            auto affinity = core->object->get_property(device_name, ov::affinity);
            ov_affinity_e* temp = new ov_affinity_e;
            *temp = static_cast<ov_affinity_e>(affinity);
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::ENUM;
            break;
        }
        case ov_property_key_e::INFERENCE_NUM_THREADS: {
            auto num = core->object->get_property(device_name, ov::inference_num_threads);
            int32_t* temp = new int32_t;
            *temp = num;
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::INT32;
            break;
        }
        case ov_property_key_e::PERFORMANCE_HINT: {
            auto perf_mode = core->object->get_property(device_name, ov::hint::performance_mode);
            ov_performance_mode_e* temp = new ov_performance_mode_e;
            *temp = static_cast<ov_performance_mode_e>(perf_mode);
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::ENUM;
            break;
        }
        case ov_property_key_e::NETWORK_NAME: {
            auto name = core->object->get_property(device_name, ov::model_name);
            char* tmp = new char[name.length() + 1];
            std::copy_n(name.begin(), name.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->cnt = name.length() + 1;
            value->type = ov_property_value_type_e::CHAR;
            break;
        }
        case ov_property_key_e::INFERENCE_PRECISION_HINT: {
            auto infer_precision = core->object->get_property(device_name, ov::hint::inference_precision);
            ov_element_type_e* temp = new ov_element_type_e;
            *temp = static_cast<ov_element_type_e>(ov::element::Type_t(infer_precision));
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::ENUM;
            break;
        }
        case ov_property_key_e::OPTIMAL_BATCH_SIZE: {
            auto batch_size = core->object->get_property(device_name, ov::optimal_batch_size);
            uint32_t* temp = new uint32_t;
            *temp = batch_size;
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::UINT32;
            break;
        }
        case ov_property_key_e::MAX_BATCH_SIZE: {
            auto batch_size = core->object->get_property(device_name, ov::max_batch_size);
            uint32_t* temp = new uint32_t;
            *temp = batch_size;
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::UINT32;
            break;
        }
        case ov_property_key_e::PERFORMANCE_HINT_NUM_REQUESTS: {
            auto num_requests = core->object->get_property(device_name, ov::hint::num_requests);
            uint32_t* temp = new uint32_t;
            *temp = num_requests;
            value->ptr = static_cast<void*>(temp);
            value->cnt = 1;
            value->type = ov_property_value_type_e::UINT32;
            break;
        }
        default:
            return ov_status_e::OUT_OF_BOUNDS;
            break;
        }
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
        for (int i = 0; i < available_devices.size(); i++) {
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
    for (int i = 0; i < devices->size; i++) {
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
        for (int i = 0; i < object.size(); i++, iter++) {
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
    for (int i = 0; i < versions->size; i++) {
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

void ov_free(const char* content) {
    if (content)
        delete content;
}
