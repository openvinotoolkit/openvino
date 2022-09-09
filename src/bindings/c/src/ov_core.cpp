// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_core.h"

#include "common.h"

//!<  Read-only property<char *> to get a string list of supported read-only properties.
const char* ov_property_key_supported_properties = "SUPPORTED_PROPERTIES";

//!<  Read-only property<char *> to get a list of available device IDs
const char* ov_property_key_available_devices = "AVAILABLE_DEVICES";

//!<  Read-only property<uint32_t> to get an unsigned integer value of optimaln
//!<  number of compiled model infer requests.
const char* ov_property_key_optimal_number_of_infer_requests = "OPTIMAL_NUMBER_OF_INFER_REQUESTS";

//!<  Read-only property<unsigned int, unsigned int, unsigned int> to provide a
//!<  hint for a range for number of async infer requests. If device supports
//!<  streams, the metric provides range for number of IRs per stream.
const char* ov_property_key_range_for_async_infer_requests = "RANGE_FOR_ASYNC_INFER_REQUESTS";

//!<  Read-only property<unsigned int, unsigned int> to provide information about a range for
//!<  streams on platforms where streams are supported
const char* ov_property_key_range_for_streams = "RANGE_FOR_STREAMS";

//!<  Read-only property<char *> to get a string value representing a full device name.
const char* ov_property_key_device_full_name = "FULL_DEVICE_NAME";

//!<  Read-only property<char *> to get a string list of capabilities options per device.
const char* ov_property_key_device_capabilities = "OPTIMIZATION_CAPABILITIES";

//!<  Read-write property<char *> to set/get the directory which will be used to store any data cached
//!<  by plugins.
const char* ov_property_key_cache_dir = "CACHE_DIR";

//!<  Read-write property<uint32_t> to set/get the number of executor logical partitions.
const char* ov_property_key_num_streams = "NUM_STREAMS";

//!<  Read-write property to set/get the name for setting CPU affinity per thread option.
const char* ov_property_key_affinity = "AFFINITY";

//!<  Read-write property<int32_t> to set/get the maximum number of threads that can be used
//!<  for inference tasks.
const char* ov_property_key_inference_num_threads = "INFERENCE_NUM_THREADS";

//!< Read-write property<ov_performance_mode_e>, it is high-level OpenVINO Performance Hints
//!< unlike low-level properties that are individual (per-device), the hints are something that
//!< every device accepts and turns into device-specific settings detail see
//!< ov_performance_mode_e to get its hint's key name
const char* ov_property_key_hint_performance_mode = "PERFORMANCE_HINT";

//!<  Read-only property<char *> to get a name of name of a model
const char* ov_property_key_model_name = "NETWORK_NAME";

//!< Read-write property<ov_element_type_e> to set the hint for device to use specified
//!< precision for inference
const char* ov_property_key_hint_inference_precision = "INFERENCE_PRECISION_HINT";

//!<  Read-only property<uint32_t> to query information optimal batch size for the given device
//!<  and the network
const char* ov_property_key_optimal_batch_size = "OPTIMAL_BATCH_SIZE";

//!<  Read-only property to get maximum batch size which does not cause performance degradation due
//!<  to memory swap impact.
const char* ov_property_key_max_batch_size = "MAX_BATCH_SIZE";

//!<  (Optional) property<uint32_t> that backs the Performance Hints by giving
//!<  additional information on how many inference requests the application will be
//!<  keeping in flight usually this value comes from the actual use-case  (e.g.
//!<  number of video-cameras, or other sources of inputs)
const char* ov_property_key_hint_num_requests = "PERFORMANCE_HINT_NUM_REQUESTS";

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
                                  const ov_properties_t* property,
                                  ov_compiled_model_t** compiled_model) {
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
                                            const ov_properties_t* property,
                                            ov_compiled_model_t** compiled_model) {
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

const std::map<ov_performance_mode_e, ov::hint::PerformanceMode> performance_mode_map = {
    {ov_performance_mode_e::UNDEFINED_MODE, ov::hint::PerformanceMode::UNDEFINED},
    {ov_performance_mode_e::THROUGHPUT, ov::hint::PerformanceMode::THROUGHPUT},
    {ov_performance_mode_e::LATENCY, ov::hint::PerformanceMode::LATENCY},
    {ov_performance_mode_e::CUMULATIVE_THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT}};

inline ov_status_e ov_core_properies_to_anymap(const ov_properties_t* properties, ov::AnyMap& dest) {
    if (!properties || properties->size <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto size = properties->size;
        for (size_t i = 0; i < size; i++) {
            auto& property = properties->list[i];
            auto& value = property.value;
            std::string key = std::string(property.key);
            if (key == ov_property_key_hint_num_requests) {
                if (value.size != 1 || value.type != ov_any_type_e::UINT32) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                uint32_t v = *(static_cast<uint32_t*>(value.ptr));
                dest.emplace(ov::hint::num_requests(v));
            } else if (key == ov_property_key_num_streams) {
                if (value.size != 1 || value.type != ov_any_type_e::UINT32) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                uint32_t v = *(static_cast<uint32_t*>(value.ptr));
                dest.emplace(ov::num_streams(v));
            } else if (key == ov_property_key_hint_performance_mode) {
                if (value.size != 1 || value.type != ov_any_type_e::ENUM) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                ov_performance_mode_e m = *(static_cast<ov_performance_mode_e*>(value.ptr));
                if (m > ov_performance_mode_e::CUMULATIVE_THROUGHPUT) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                auto v = performance_mode_map.at(m);
                dest.emplace(ov::hint::performance_mode(v));
            } else if (key == ov_property_key_affinity) {
                if (value.size != 1 || value.type != ov_any_type_e::ENUM) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                ov_affinity_e v = *(static_cast<ov_affinity_e*>(value.ptr));
                if (v < ov_affinity_e::NONE || v > ov_affinity_e::HYBRID_AWARE) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                ov::Affinity affinity = static_cast<ov::Affinity>(v);
                dest.emplace(ov::affinity(affinity));
            } else if (key == ov_property_key_inference_num_threads) {
                if (value.size != 1 || value.type != ov_any_type_e::INT32) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                int32_t v = *(static_cast<int32_t*>(value.ptr));
                dest.emplace(ov::inference_num_threads(v));
            } else if (key == ov_property_key_hint_inference_precision) {
                if (value.size != 1 || value.type != ov_any_type_e::ENUM) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                ov_element_type_e v = *(static_cast<ov_element_type_e*>(value.ptr));
                if (v > ov_element_type_e::U64) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                ov::element::Type type(static_cast<ov::element::Type_t>(v));
                dest.emplace(ov::hint::inference_precision(type));
            } else if (key == ov_property_key_cache_dir) {
                if (value.size < 1 || value.type != ov_any_type_e::CHAR) {
                    return ov_status_e::INVALID_C_PARAM;
                }
                char* dir = static_cast<char*>(value.ptr);
                dest.emplace(ov::cache_dir(std::string(dir)));
            } else {
                return ov_status_e::NOT_IMPLEMENTED;
            }
        }  //  end of for
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_set_property(const ov_core_t* core, const char* device_name, const ov_properties_t* property) {
    if (!core || !property) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap dest;
        auto ret = ov_core_properies_to_anymap(property, dest);
        if (ret != ov_status_e::OK) {
            return ret;
        }

        if (device_name) {
            core->object->set_property(device_name, dest);
        } else {
            core->object->set_property(dest);
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_property(const ov_core_t* core, const char* device_name, const char* key, ov_any_t* value) {
    if (!core || !value || !key) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string _key = std::string(key);
        if (_key == ov_property_key_supported_properties) {
            auto supported_properties = core->object->get_property(device_name, ov::supported_properties);
            std::string tmp_s;
            for (const auto& i : supported_properties) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* tmp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->size = tmp_s.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else if (_key == ov_property_key_available_devices) {
            auto available_devices = core->object->get_property(device_name, ov::available_devices);
            std::string tmp_s;
            for (const auto& i : available_devices) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* tmp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->size = tmp_s.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else if (_key == ov_property_key_optimal_number_of_infer_requests) {
            auto optimal_number_of_infer_requests =
                core->object->get_property(device_name, ov::optimal_number_of_infer_requests);
            uint32_t* temp = new uint32_t;
            *temp = optimal_number_of_infer_requests;
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::UINT32;
        } else if (_key == ov_property_key_range_for_async_infer_requests) {
            auto range = core->object->get_property(device_name, ov::range_for_async_infer_requests);
            uint32_t* temp = new uint32_t[3];
            temp[0] = std::get<0>(range);
            temp[1] = std::get<1>(range);
            temp[2] = std::get<2>(range);
            value->ptr = static_cast<void*>(temp);
            value->size = 3;
            value->type = ov_any_type_e::UINT32;
        } else if (_key == ov_property_key_range_for_streams) {
            auto range = core->object->get_property(device_name, ov::range_for_streams);
            uint32_t* temp = new uint32_t[2];
            temp[0] = std::get<0>(range);
            temp[1] = std::get<1>(range);
            value->ptr = static_cast<void*>(temp);
            value->size = 2;
            value->type = ov_any_type_e::UINT32;
        } else if (_key == ov_property_key_device_full_name) {
            auto name = core->object->get_property(device_name, ov::device::full_name);
            char* tmp = new char[name.length() + 1];
            std::copy_n(name.begin(), name.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->size = name.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else if (_key == ov_property_key_device_capabilities) {
            auto capabilities = core->object->get_property(device_name, ov::device::capabilities);
            std::string tmp_s;
            for (const auto& i : capabilities) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* tmp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->size = tmp_s.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else if (_key == ov_property_key_cache_dir) {
            auto dir = core->object->get_property(device_name, ov::cache_dir);
            char* tmp = new char[dir.length() + 1];
            std::copy_n(dir.begin(), dir.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->size = dir.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else if (_key == ov_property_key_num_streams) {
            auto num = core->object->get_property(device_name, ov::num_streams);
            int32_t* temp = new int32_t;
            *temp = num.num;
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::INT32;
        } else if (_key == ov_property_key_affinity) {
            auto affinity = core->object->get_property(device_name, ov::affinity);
            ov_affinity_e* temp = new ov_affinity_e;
            *temp = static_cast<ov_affinity_e>(affinity);
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::ENUM;
        } else if (_key == ov_property_key_inference_num_threads) {
            auto num = core->object->get_property(device_name, ov::inference_num_threads);
            int32_t* temp = new int32_t;
            *temp = num;
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::INT32;
        } else if (_key == ov_property_key_hint_performance_mode) {
            auto perf_mode = core->object->get_property(device_name, ov::hint::performance_mode);
            ov_performance_mode_e* temp = new ov_performance_mode_e;
            *temp = static_cast<ov_performance_mode_e>(perf_mode);
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::ENUM;
        } else if (_key == ov_property_key_model_name) {
            auto name = core->object->get_property(device_name, ov::model_name);
            char* tmp = new char[name.length() + 1];
            std::copy_n(name.begin(), name.length() + 1, tmp);
            value->ptr = static_cast<void*>(tmp);
            value->size = name.length() + 1;
            value->type = ov_any_type_e::CHAR;
        } else if (_key == ov_property_key_hint_inference_precision) {
            auto infer_precision = core->object->get_property(device_name, ov::hint::inference_precision);
            ov_element_type_e* temp = new ov_element_type_e;
            *temp = static_cast<ov_element_type_e>(ov::element::Type_t(infer_precision));
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::ENUM;
        } else if (_key == ov_property_key_optimal_batch_size) {
            auto batch_size = core->object->get_property(device_name, ov::optimal_batch_size);
            uint32_t* temp = new uint32_t;
            *temp = batch_size;
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::UINT32;
        } else if (_key == ov_property_key_max_batch_size) {
            auto batch_size = core->object->get_property(device_name, ov::max_batch_size);
            uint32_t* temp = new uint32_t;
            *temp = batch_size;
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::UINT32;
        } else if (_key == ov_property_key_hint_num_requests) {
            auto num_requests = core->object->get_property(device_name, ov::hint::num_requests);
            uint32_t* temp = new uint32_t;
            *temp = num_requests;
            value->ptr = static_cast<void*>(temp);
            value->size = 1;
            value->type = ov_any_type_e::UINT32;
        } else {
            return ov_status_e::OUT_OF_BOUNDS;
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
