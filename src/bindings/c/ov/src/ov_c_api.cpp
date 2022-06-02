// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <string>
#include <utility>
#include <cassert>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <memory>
#include <streambuf>
#include <fstream>

#include "c_api/ov_c_api.h"
#include "openvino/openvino.hpp"

struct ov_core {
    std::shared_ptr<ov::Core> object;
};

struct ov_node {
    std::shared_ptr<ov::Node> object;
};

struct ov_output_node {
    std::shared_ptr<ov::Output<const ov::Node>> object;
};

struct ov_model {
    std::shared_ptr<ov::Model> object;
};

struct ov_preprocess {
    std::shared_ptr<ov::preprocess::PrePostProcessor> object;
};

struct ov_preprocess_input_info {
    ov::preprocess::InputInfo* object;
};

struct ov_preprocess_input_tensor_info {
    ov::preprocess::InputTensorInfo* object;
};

struct ov_preprocess_output_info {
    ov::preprocess::OutputInfo* object;
};

struct ov_preprocess_output_tensor_info {
    ov::preprocess::OutputTensorInfo* object;
};

struct ov_preprocess_input_model_info {
    ov::preprocess::InputModelInfo* object;
};

struct ov_preprocess_input_process_steps {
    ov::preprocess::PreProcessSteps* object;
};

struct ov_compiled_model {
    std::shared_ptr<ov::CompiledModel> object;
};

struct ov_infer_request {
    std::shared_ptr<ov::InferRequest> object;
};

struct ov_tensor {
    std::shared_ptr<ov::Tensor> object;
};

/**
 * @variable global value for error info
 */
char const* error_infos[] = {
    "no error.",
    "general error!",
    "not implement!",
    "network load failed!",
    "input parameter mismatch!",
    "cannot find the value!",
    "out of bounds!",
    "run with unexpected error!",
    "request is busy now!",
    "result is not ready now!",
    "allocated failed!",
    "inference start with error!",
    "network is not ready now!",
    "inference is canceled!",
    "unknown value!",
};

/**
 * @struct mem_stringbuf
 * @brief This struct puts memory buffer to stringbuf.
 */
struct mem_stringbuf : std::streambuf {
    mem_stringbuf(const char *buffer, size_t sz) {
        char * bptr(const_cast<char *>(buffer));
        setg(bptr, bptr, bptr + sz);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in) override {
        switch (dir) {
            case std::ios_base::beg:
                setg(eback(), eback() + off, egptr());
                break;
            case std::ios_base::end:
                setg(eback(), egptr() + off, egptr());
                break;
            case std::ios_base::cur:
                setg(eback(), gptr() + off, egptr());
                break;
            default:
                return pos_type(off_type(-1));
        }
        return (gptr() < eback() || gptr() > egptr()) ? pos_type(off_type(-1)) : pos_type(gptr() - eback());
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

/**
 * @struct mem_istream
 * @brief This struct puts stringbuf buffer to istream.
 */
struct mem_istream: virtual mem_stringbuf, std::istream {
    mem_istream(const char * buffer, size_t sz) : mem_stringbuf(buffer, sz), std::istream(static_cast<std::streambuf *>(this)) {
    }
};

std::map<ov_performance_mode_e, ov::hint::PerformanceMode> performance_mode_map = {
    {ov_performance_mode_e::UNDEFINED_MODE, ov::hint::PerformanceMode::UNDEFINED},
    {ov_performance_mode_e::THROUGHPUT, ov::hint::PerformanceMode::THROUGHPUT},
    {ov_performance_mode_e::LATENCY, ov::hint::PerformanceMode::LATENCY},
    {ov_performance_mode_e::CUMULATIVE_THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT}};

std::map<ov_preprocess_resize_algorithm_e, ov::preprocess::ResizeAlgorithm> resize_algorithm_map = {
    {ov_preprocess_resize_algorithm_e::RESIZE_CUBIC, ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC},
    {ov_preprocess_resize_algorithm_e::RESIZE_LINEAR, ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR},
    {ov_preprocess_resize_algorithm_e::RESIZE_NEAREST, ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST}};

std::map<ov_element_type_e, ov::element::Type> element_type_map = {
    {ov_element_type_e::UNDEFINED, ov::element::undefined},
    {ov_element_type_e::DYNAMIC, ov::element::dynamic},
    {ov_element_type_e::BOOLEAN, ov::element::boolean},
    {ov_element_type_e::BF16, ov::element::bf16},
    {ov_element_type_e::F16, ov::element::f16},
    {ov_element_type_e::F32, ov::element::f32},
    {ov_element_type_e::F64, ov::element::f64},
    {ov_element_type_e::I4, ov::element::i4},
    {ov_element_type_e::I8, ov::element::i8},
    {ov_element_type_e::I16, ov::element::i16},
    {ov_element_type_e::I32, ov::element::i32},
    {ov_element_type_e::I64, ov::element::i64},
    {ov_element_type_e::U1, ov::element::u1},
    {ov_element_type_e::U4, ov::element::u4},
    {ov_element_type_e::U8, ov::element::u8},
    {ov_element_type_e::U16, ov::element::u16},
    {ov_element_type_e::U32, ov::element::u32},
    {ov_element_type_e::U64, ov::element::u64}};

ov_element_type_e find_ov_element_type_e(ov::element::Type type) {
    for (auto iter = element_type_map.begin(); iter != element_type_map.end(); iter++) {
        if (iter->second == type) {
            return iter->first;
        }
    }
    return ov_element_type_e::UNDEFINED;
}

#define GET_OV_ELEMENT_TYPE(a) element_type_map[a]
#define GET_CAPI_ELEMENT_TYPE(a) find_ov_element_type_e(a)

#define CATCH_OV_EXCEPTION(StatusCode, ExceptionType) catch (const InferenceEngine::ExceptionType&) {return ov_status_e::StatusCode;}

#define CATCH_OV_EXCEPTIONS                                         \
        CATCH_OV_EXCEPTION(GENERAL_ERROR, GeneralError)             \
        CATCH_OV_EXCEPTION(NOT_IMPLEMENTED, NotImplemented)         \
        CATCH_OV_EXCEPTION(NETWORK_NOT_LOADED, NetworkNotLoaded)    \
        CATCH_OV_EXCEPTION(PARAMETER_MISMATCH, ParameterMismatch)   \
        CATCH_OV_EXCEPTION(NOT_FOUND, NotFound)                     \
        CATCH_OV_EXCEPTION(OUT_OF_BOUNDS, OutOfBounds)              \
        CATCH_OV_EXCEPTION(UNEXPECTED, Unexpected)                  \
        CATCH_OV_EXCEPTION(REQUEST_BUSY, RequestBusy)               \
        CATCH_OV_EXCEPTION(RESULT_NOT_READY, ResultNotReady)        \
        CATCH_OV_EXCEPTION(NOT_ALLOCATED, NotAllocated)             \
        CATCH_OV_EXCEPTION(INFER_NOT_STARTED, InferNotStarted)      \
        CATCH_OV_EXCEPTION(NETWORK_NOT_READ, NetworkNotRead)        \
        CATCH_OV_EXCEPTION(INFER_CANCELLED, InferCancelled)         \
        catch (...) {return ov_status_e::UNEXPECTED;}

char* str_to_char_array(const std::string& str) {
    char *char_array = new char[str.length() + 1];
    std::copy_n(str.begin(), str.length() + 1, char_array);
    return char_array;
}

const char* ov_get_error_info(ov_status_e status) {
    if (status > ov_status_e::UNKNOWN_ERROR)
        return error_infos[ov_status_e::UNKNOWN_ERROR];
    return error_infos[status];
}

ov_status_e ov_get_version(ov_version_t *version) {
    if (!version) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        ov::Version object = ov::get_openvino_version();

        std::string version_builderNumber = object.buildNumber;
        version->buildNumber = str_to_char_array(version_builderNumber);

        std::string version_description = object.description;
        version->description = str_to_char_array(version_description);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_version_free(ov_version_t *version) {
    if (!version) {
        return;
    }
    delete[] version->buildNumber;
    version->buildNumber = nullptr;
    delete[] version->description;
    version->description = nullptr;
}

ov_status_e ov_core_create(const char *xml_config_file, ov_core_t **core) {
    if (!core || !xml_config_file) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *core = new ov_core_t;
        (*core)->object = std::make_shared<ov::Core>(xml_config_file);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_core_free(ov_core_t *core) {
        delete core;
}

ov_status_e ov_core_read_model(const ov_core_t *core,
                        const char *model_path,
                        const char *bin_path,
                        ov_model_t **model) {
    if (!core || !model_path || !model) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        std::string bin = "";
        if (bin_path) {
            bin = bin_path;
        }
        *model = new ov_model_t;
        (*model)->object = core->object->read_model(model_path, bin);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_read_model_from_memory(const ov_core_t *core,
                                    const char *model_str,
                                    const ov_tensor_t *weights,
                                    ov_model_t **model) {
    if (!core || !model_str || !weights || !model) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *model = new ov_model_t;
        (*model)->object = core->object->read_model(model_str, *(weights->object));
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_model_free(ov_model_t *model) {
    delete model;
}

ov_status_e ov_core_compile_model(const ov_core_t* core,
                            const ov_model_t* model,
                            const char* device_name,
                            ov_compiled_model_t **compiled_model,
                            const ov_property_t* property) {
    if (!core || !model || !compiled_model) {
        return ov_status_e::GENERAL_ERROR;
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
        *compiled_model = new ov_compiled_model_t;
        (*compiled_model)->object = std::make_shared<ov::CompiledModel>(std::move(object));
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_compile_model_from_file(const ov_core_t* core,
                                    const char* model_path,
                                    const char* device_name,
                                    ov_compiled_model_t **compiled_model,
                                    const ov_property_t* property) {
    if (!core || !model_path || !compiled_model) {
        return ov_status_e::GENERAL_ERROR;
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
        *compiled_model = new ov_compiled_model_t;
        (*compiled_model)->object = std::make_shared<ov::CompiledModel>(std::move(object));
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_compiled_model_free(ov_compiled_model_t *compiled_model) {
    delete compiled_model;
}

ov_status_e ov_core_set_property(const ov_core_t* core,
                            const char* device_name,
                            const ov_property_t* property) {
    if (!core || !property) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        ov::AnyMap config;
        while (property) {
            switch (property->key) {
            case ov_property_key_e::PERFORMANCE_HINT_NUM_REQUESTS:
                config.emplace(ov::hint::num_requests(property->value.value_u));
                break;
            case ov_property_key_e::NUM_STREAMS:
                config.emplace(ov::num_streams(property->value.value_u));
                break;
            case ov_property_key_e::PERFORMANCE_HINT:
                config.emplace(ov::hint::performance_mode(performance_mode_map[property->value.value_performance_mode]));
                break;
            default:
                break;
            }
            property = property->next;
        }

        if (device_name) {
            core->object->set_property(device_name, config);
        } else {
            core->object->set_property(config);
        }
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_property(const ov_core_t* core, const char* device_name,
                            const ov_property_key_e property_key,
                            ov_property_value* property_value) {
    if (!core || !device_name || !property_value) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        switch (property_key) {
        case ov_property_key_e::SUPPORTED_PROPERTIES:
        {
            auto supported_properties = core->object->get_property(device_name, ov::supported_properties);
            std::string tmp_s;
            for (const auto& i : supported_properties) {
                tmp_s = tmp_s + "\n" + i;
            }
            if (tmp_s.length() + 1 > 512) {
                return ov_status_e::GENERAL_ERROR;
            }
            std::copy_n(tmp_s.begin(), tmp_s.length() + 1, property_value->value_s);
            break;
        }
        default:
            break;
        }
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_available_devices(const ov_core_t* core, ov_available_devices_t* devices) {
    if (!core) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto available_devices = core->object->get_available_devices();
        devices->num_devices = available_devices.size();
        auto tmp_devices(new char*[available_devices.size()]);
        for (int i = 0; i < available_devices.size(); i++) {
            tmp_devices[i] = str_to_char_array(available_devices[i]);
        }
        devices->devices = tmp_devices;
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_available_devices_free(ov_available_devices_t* devices) {
    if (!devices) {
        return;
    }
    for (int i = 0; i < devices->num_devices; i++) {
        if (devices->devices[i]) {
            delete [] devices->devices[i];
        }
    }
    delete [] devices->devices;
    devices->devices = nullptr;
    devices->num_devices = 0;
}

ov_status_e ov_core_import_model(const ov_core_t* core,
                            const char *content,
                            const size_t content_size,
                            const char* device_name,
                            ov_compiled_model_t **compiled_model) {
    if (!core || !content || !device_name || !compiled_model) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        mem_istream model_stream(content, content_size);
        *compiled_model = new ov_compiled_model_t;
        auto object = core->object->import_model(model_stream, device_name);
        (*compiled_model)->object = std::make_shared<ov::CompiledModel>(std::move(object));
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_core_get_versions(const ov_core_t* core,
                            const char* device_name,
                            ov_core_version_list_t *versions) {
    if (!core || !device_name || !versions) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto object = core->object->get_versions(device_name);
        if (object.empty()) {
            return ov_status_e::NOT_FOUND;
        }
        versions->num_vers = object.size();
        auto tmp_versions(new ov_core_version_t[object.size()]);
        auto iter = object.cbegin();
        for (int i = 0; i < object.size(); i++, iter++) {
            const auto& tmp_version_name = iter->first;
            tmp_versions[i].device_name = str_to_char_array(tmp_version_name);

            const auto tmp_version_build_number = iter->second.buildNumber;
            tmp_versions[i].buildNumber = str_to_char_array(tmp_version_build_number);

            const auto tmp_version_description = iter->second.description;
            tmp_versions[i].description = str_to_char_array(tmp_version_description);
        }
        versions->versions = tmp_versions;
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_core_versions_free(ov_core_version_list_t *versions) {
    if (!versions) {
        return;
    }
    for (int i = 0; i < versions->num_vers; i++) {
        delete[] versions->versions[i].device_name;
        delete[] versions->versions[i].buildNumber;
        delete[] versions->versions[i].description;
    }
    delete[] versions->versions;
    versions->versions = nullptr;
}

ov_status_e ov_model_get_outputs(const ov_model_t* model, ov_output_node_list_t *output_nodes) {
    if (!model || !output_nodes) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto results = std::const_pointer_cast<const ov::Model>(model->object)->outputs();
        output_nodes->num = results.size();
        auto tmp_output_nodes(new ov_output_node_t[output_nodes->num]);

        for (size_t i = 0; i < output_nodes->num; i++) {
            tmp_output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(results[i]));
        }
        output_nodes->output_nodes = tmp_output_nodes;
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_get_inputs(const ov_model_t* model, ov_output_node_list_t *input_nodes) {
    if (!model || !input_nodes) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto results = std::const_pointer_cast<const ov::Model>(model->object)->inputs();
        input_nodes->num = results.size();
        auto tmp_output_nodes(new ov_output_node_t[input_nodes->num]);

        for (size_t i = 0; i < input_nodes->num; i++) {
            tmp_output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(results[i]));
        }
        input_nodes->output_nodes = tmp_output_nodes;
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_node_get_tensor_name(ov_output_node_list_t* nodes, size_t idx,
                                    char** tensor_name) {
    if (!nodes || !tensor_name || idx >= nodes->num) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *tensor_name = str_to_char_array(nodes->output_nodes[idx].object->get_any_name());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_tensor_shape(ov_output_node_list_t* nodes, size_t idx,
                                    ov_shape_t* tensor_shape) {
    if (!nodes || idx >= nodes->num) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        auto shape = nodes->output_nodes[idx].object->get_shape();
        if (shape.size() > MAX_DIMENSION) {
            return ov_status_e::GENERAL_ERROR;
        }
        tensor_shape->ranks = shape.size();
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_node_get_tensor_type(ov_output_node_list_t* nodes, size_t idx,
                                    ov_element_type_e *tensor_type) {
    if (!nodes || idx >= nodes->num) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        auto type = (ov::element::Type_t)nodes->output_nodes[idx].object->get_element_type();
        *tensor_type = (ov_element_type_e)type;
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_model_get_input_by_name(const ov_model_t* model,
                                const char* tensor_name,
                                ov_output_node_t **input_node) {
    if (!model || !tensor_name || !input_node) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(tensor_name);
        *input_node = new ov_output_node_t;
        (*input_node)->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_get_input_by_id(const ov_model_t* model,
                                const size_t index,
                                ov_output_node_t **input_node) {
    if (!model || index < 0 || !input_node) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(index);
        *input_node = new ov_output_node_t;
        (*input_node)->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

bool ov_model_is_dynamic(const ov_model_t* model) {
    if (!model) {
        printf("[ERROR] The model is NULL!!!\n");
        return false;
    }
    return model->object->is_dynamic();
}

std::vector<std::string> split(std::string s, std::vector<std::string> delimiter) {
    size_t s_start = 0, s_end;
    std::string token;
    std::vector<std::string> result;
    for (int i = 0; i < s.size(); ++i) {
        for (auto delim : delimiter) {
            if ((s_end = s.find(delim, s_start)) != std::string::npos) {
                token = s.substr(s_start, s_end - s_start);
                s_start = s_end + delim.size();
                result.push_back(token);
            }
        }
    }
    result.push_back(s.substr(s_start));
    return result;
}

ov_status_e ov_partial_shape_init(ov_partial_shape_t* partial_shape, const char* str) {
    if (!partial_shape || !str) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        std::string s = str;
        std::vector<std::string> delimiter = {",", ":", "+", ";"};
        std::vector<std::string> result = split(s, delimiter);
        partial_shape->ranks = result.size();
        std::vector<char*> res;
        for (auto i : result) {
            res.push_back(str_to_char_array(i));
        }
        std::copy_n(res.begin(), res.size(), partial_shape->dims);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

const char* ov_partial_shape_parse(ov_partial_shape_t* partial_shape) {
    if (!partial_shape) {
        return "error";
    }
    std::string str = "";
    const char* res;
    for (int i = 0; i < partial_shape->ranks; ++i) {
        std::string tmp = partial_shape->dims[i];
        str += tmp;
        if (i != partial_shape->ranks - 1)
            str += ",";
    }
    res = str_to_char_array(str);
    return res;
}

ov_status_e ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape) {
    if (!partial_shape || !shape) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        std::vector<size_t> tmp_shape;
        for (int i = 0; i < partial_shape->ranks; ++i) {
            std::string dim = partial_shape->dims[i];
            if (dim == "?" || dim == "-1" || dim.find("..") != std::string::npos) {
                return ov_status_e::GENERAL_ERROR;
            } else {
                tmp_shape.push_back(std::stoi(dim));
            }
        }
        std::copy_n(tmp_shape.begin(), tmp_shape.size(), shape->dims);
        shape->ranks = partial_shape->ranks;
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape(const ov_model_t* model,
                        const char* tensor_name,
                        const ov_partial_shape_t partial_shape) {
    if (!model || !tensor_name) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        std::vector<ngraph::Dimension> shape;
        for (int i = 0; i < MAX_DIMENSION; i++) {
            if (partial_shape.dims[i] == nullptr)
                break;
            std::string dim = partial_shape.dims[i];
            if (dim == "?" || dim == "-1") {
                shape.push_back(ov::Dimension::dynamic());
            } else {
                const std::string range_divider = "..";
                size_t range_index = dim.find(range_divider);
                if (range_index != std::string::npos) {
                    std::string min = dim.substr(0, range_index);
                    std::string max = dim.substr(range_index + range_divider.length());
                    shape.emplace_back(min.empty() ? 0 : std::stoi(min),
                                    max.empty() ? ngraph::Interval::s_max : std::stoi(max));
                } else {
                    shape.emplace_back(std::stoi(dim));
                }
            }
        }

        std::map<std::string, ov::PartialShape> const_pshape;
        const_pshape[tensor_name] = shape;
        model->object->reshape(const_pshape);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name) {
    if (!model || !friendly_name) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto& result = model->object->get_friendly_name();
        *friendly_name = str_to_char_array(result);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_output_node_list_free(ov_output_node_list_t *output_nodes) {
    if (output_nodes) {
        delete[] output_nodes->output_nodes;
        output_nodes->output_nodes = nullptr;
    }
}

void ov_output_node_free(ov_output_node_t *output_node) {
    delete output_node;
}

void ov_free(char *content) {
    delete content;
}

void ov_partial_shape_free(ov_partial_shape_t* partial_shape) {
    if (partial_shape) {
        for (int i = 0; i < partial_shape->ranks; i++) {
            if (partial_shape->dims[i]) {
                delete [] partial_shape->dims[i];
            }
        }
        partial_shape->ranks = 0;
    }
}

void ov_shape_free(ov_shape_t* shape) {
    delete shape;
}

ov_status_e ov_preprocess_create(const ov_model_t* model,
                            ov_preprocess_t **preprocess) {
    if (!model || !preprocess) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess = new ov_preprocess_t;
        (*preprocess)->object = std::make_shared<ov::preprocess::PrePostProcessor>(model->object);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_free(ov_preprocess_t *preprocess) {
    delete preprocess;
}

ov_status_e ov_preprocess_get_input_info(const ov_preprocess_t* preprocess,
                                    ov_preprocess_input_info_t **preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_input_info = new ov_preprocess_input_info_t;
        (*preprocess_input_info)->object = &(preprocess->object->input());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_get_input_info_by_name(const ov_preprocess_t* preprocess,
                                        const char* tensor_name,
                                        ov_preprocess_input_info_t **preprocess_input_info) {
    if (!preprocess || !tensor_name || !preprocess_input_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_input_info = new ov_preprocess_input_info_t;
        (*preprocess_input_info)->object = &(preprocess->object->input(tensor_name));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_get_input_info_by_index(const ov_preprocess_t* preprocess,
                                            const size_t tensor_index,
                                            ov_preprocess_input_info_t **preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_input_info = new ov_preprocess_input_info_t;
        (*preprocess_input_info)->object = &(preprocess->object->input(tensor_index));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_info_free(ov_preprocess_input_info_t *preprocess_input_info) {
    delete preprocess_input_info;
}

ov_status_e ov_preprocess_input_get_tensor_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                            ov_preprocess_input_tensor_info_t **preprocess_input_tensor_info) {
    if (!preprocess_input_info || !preprocess_input_tensor_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_input_tensor_info = new ov_preprocess_input_tensor_info_t;
        (*preprocess_input_tensor_info)->object = &(preprocess_input_info->object->tensor());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_tensor_info_free(ov_preprocess_input_tensor_info_t *preprocess_input_tensor_info) {
    delete preprocess_input_tensor_info;
}

ov_status_e ov_preprocess_input_get_preprocess_steps(const ov_preprocess_input_info_t* preprocess_input_info,
                                                ov_preprocess_input_process_steps_t **preprocess_input_steps) {
    if (!preprocess_input_info || !preprocess_input_steps) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_input_steps = new ov_preprocess_input_process_steps_t;
        (*preprocess_input_steps)->object = &(preprocess_input_info->object->preprocess());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_process_steps_free(ov_preprocess_input_process_steps_t *preprocess_input_process_steps) {
    delete preprocess_input_process_steps;
}

ov_status_e ov_preprocess_input_resize(ov_preprocess_input_process_steps_t* preprocess_input_process_steps,
                                    const ov_preprocess_resize_algorithm_e resize_algorithm) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        preprocess_input_process_steps->object->resize(resize_algorithm_map[resize_algorithm]);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_input_tensor_info_set_element_type(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                        const ov_element_type_e element_type) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        preprocess_input_tensor_info->object->set_element_type(GET_OV_ELEMENT_TYPE(element_type));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_input_tensor_info_set_tensor(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                const ov_tensor_t* tensor) {
    if (!preprocess_input_tensor_info || !tensor) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        preprocess_input_tensor_info->object->set_from(*(tensor->object));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                const ov_layout_t layout) {
    if (!preprocess_input_tensor_info || !layout) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        ov::Layout tmp_layout(std::string(layout, 4));
        preprocess_input_tensor_info->object->set_layout(tmp_layout);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_get_output_info(const ov_preprocess_t* preprocess,
                                    ov_preprocess_output_info_t **preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_output_info = new ov_preprocess_output_info_t;
        (*preprocess_output_info)->object = &(preprocess->object->output());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_get_output_info_by_index(const ov_preprocess_t* preprocess,
                                            const size_t tensor_index,
                                            ov_preprocess_output_info_t **preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_output_info = new ov_preprocess_output_info_t;
        (*preprocess_output_info)->object = &(preprocess->object->output(tensor_index));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_get_output_info_by_name(const ov_preprocess_t* preprocess,
                                            const char* tensor_name,
                                            ov_preprocess_output_info_t **preprocess_output_info) {
    if (!preprocess || !tensor_name || !preprocess_output_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_output_info = new ov_preprocess_output_info_t;
        (*preprocess_output_info)->object = &(preprocess->object->output(tensor_name));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_output_info_free(ov_preprocess_output_info_t *preprocess_output_info) {
    delete preprocess_output_info;
}

ov_status_e ov_preprocess_output_get_tensor_info(ov_preprocess_output_info_t* preprocess_output_info,
                                            ov_preprocess_output_tensor_info_t **preprocess_output_tensor_info) {
    if (!preprocess_output_info || !preprocess_output_tensor_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_output_tensor_info = new ov_preprocess_output_tensor_info_t;
        (*preprocess_output_tensor_info)->object = &(preprocess_output_info->object->tensor());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_output_tensor_info_free(ov_preprocess_output_tensor_info_t *preprocess_output_tensor_info) {
    delete preprocess_output_tensor_info;
}

ov_status_e ov_preprocess_output_set_element_type(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info,
                                            const ov_element_type_e element_type) {
    if (!preprocess_output_tensor_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        preprocess_output_tensor_info->object->set_element_type(GET_OV_ELEMENT_TYPE(element_type));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_input_get_model_info(ov_preprocess_input_info_t* preprocess_input_info,
                                        ov_preprocess_input_model_info_t **preprocess_input_model_info) {
    if (!preprocess_input_info || !preprocess_input_model_info) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *preprocess_input_model_info = new ov_preprocess_input_model_info_t;
        (*preprocess_input_model_info)->object = &(preprocess_input_info->object->model());
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_input_model_info_free(ov_preprocess_input_model_info_t *preprocess_input_model_info) {
    delete preprocess_input_model_info;
}

ov_status_e ov_preprocess_input_model_set_layout(ov_preprocess_input_model_info_t* preprocess_input_model_info,
                                            const ov_layout_t layout) {
    if (!preprocess_input_model_info || !layout) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        ov::Layout tmp_layout(std::string(layout, 4));
        preprocess_input_model_info->object->set_layout(tmp_layout);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_build(const ov_preprocess_t* preprocess,
                            ov_model_t **model) {
    if (!preprocess || !model) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *model = new ov_model_t;
        (*model)->object = preprocess->object->build();
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov::AnyMap property_map(const ov_property_t *property) {
    ov::AnyMap config;
    while (property) {
        switch (property->key) {
        case ov_property_key_e::PERFORMANCE_HINT_NUM_REQUESTS:
            config.emplace(ov::hint::num_requests(property->value.value_u));
            break;
        case ov_property_key_e::NUM_STREAMS:
            config.emplace(ov::num_streams(property->value.value_u));
            break;
        case ov_property_key_e::PERFORMANCE_HINT:
            config.emplace(ov::hint::performance_mode(performance_mode_map[property->value.value_performance_mode]));
            break;
        default:
            break;
        }
        property = property->next;
    }
    return config;
}

ov_status_e ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model,
                                                ov_model_t **model) {
    if (!compiled_model || !model) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *model = new ov_model_t;
        auto runtime_model = compiled_model->object->get_runtime_model();
        (*model)->object = std::const_pointer_cast<ov::Model>(std::move(runtime_model));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_inputs(const ov_compiled_model_t* compiled_model,
                                        ov_output_node_list_t *input_nodes) {
    if (!compiled_model || !input_nodes) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        auto inputs = compiled_model->object->inputs();
        int num = inputs.size();
        input_nodes->num = num;
        input_nodes->output_nodes = new ov_output_node_t[num];
        for (int i = 0; i < num; i++) {
            input_nodes->output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(inputs[i]));
        }
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_outputs(const ov_compiled_model_t* compiled_model,
                                        ov_output_node_list_t *output_nodes) {
    if (!compiled_model || !output_nodes) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        auto outputs = compiled_model->object->outputs();
        int num = outputs.size();
        output_nodes->num = num;
        output_nodes->output_nodes = new ov_output_node_t[num];
        for (int i = 0; i < num; i++) {
            output_nodes->output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(outputs[i]));
        }
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model,
                                                    ov_infer_request_t **infer_request) {
    if (!compiled_model || !infer_request) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *infer_request = new ov_infer_request_t;
        auto inferReq = compiled_model->object->create_infer_request();
        (*infer_request)->object = std::make_shared<ov::InferRequest>(std::move(inferReq));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model,
                                            const ov_property_t* property) {
    if (!compiled_model || !property) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        ov::AnyMap config = property_map(property);
        compiled_model->object->set_property(config);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                                const ov_property_key_e property_name,
                                ov_property_value* property_value) {
    if (!compiled_model || !property_value) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        switch (property_name) {
        case ov_property_key_e::SUPPORTED_PROPERTIES:
        {
            auto supported_properties = compiled_model->object->get_property(ov::supported_properties);
            std::string tmp_s;
            for (const auto& i : supported_properties) {
                tmp_s = tmp_s + "\n" + i;
            }
            if (tmp_s.length() + 1 > 256) {
                return ov_status_e::GENERAL_ERROR;
            }
            std::copy_n(tmp_s.c_str(), tmp_s.length() + 1, property_value->value_s);
            break;
        }
        default:
            break;
        }
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_export(const ov_compiled_model_t* compiled_model,
                                const char* export_model_path) {
    if (!compiled_model || !export_model_path) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        std::ofstream model_file(export_model_path, std::ios::out | std::ios::binary);
        if (model_file.is_open()) {
            compiled_model->object->export_model(model_file);
        } else {
            return ov_status_e::GENERAL_ERROR;
        }
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_infer_request_free(ov_infer_request_t *infer_request) {
    delete infer_request;
}

ov_status_e ov_infer_request_set_tensor(ov_infer_request_t *infer_request,
                                const char* tensor_name, const ov_tensor_t *tensor) {
    if (!infer_request || !tensor_name || !tensor) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        infer_request->object->set_tensor(tensor_name, *tensor->object);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request,
                                size_t idx, const ov_tensor_t* tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        infer_request->object->set_input_tensor(idx, *tensor->object);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_tensor(const ov_infer_request_t* infer_request,
                                const char* tensor_name, ov_tensor_t **tensor) {
    if (!infer_request || !tensor_name || !tensor) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *tensor = new ov_tensor_t;
        ov::Tensor tensor_get = infer_request->object->get_tensor(tensor_name);
        (*tensor)->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_out_tensor(const ov_infer_request_t* infer_request,
                                size_t idx, ov_tensor_t **tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *tensor = new ov_tensor_t;
        ov::Tensor tensor_get = infer_request->object->get_output_tensor(idx);
        (*tensor)->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_infer(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        infer_request->object->infer();
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_cancel(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        infer_request->object->cancel();
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_start_async(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        infer_request->object->start_async();
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_wait(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        infer_request->object->wait();
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_callback(ov_infer_request_t* infer_request,
                                        const ov_call_back_t* callback) {
    if (!infer_request || !callback) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        auto func = [callback](std::exception_ptr ex) {
            callback->callback_func(callback->args);
        };
        infer_request->object->set_callback(func);
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_profiling_info(ov_infer_request_t* infer_request,
                                            ov_profiling_info_list_t* profiling_infos) {
    if (!infer_request || !profiling_infos) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        auto infos = infer_request->object->get_profiling_info();
        int num = infos.size();
        profiling_infos->num = num;
        ov_profiling_info_t *profiling_info_arr = new ov_profiling_info_t[num];
        for (int i = 0; i < num; i++) {
            profiling_info_arr[i].status = (ov_profiling_info_t::Status)infos[i].status;
            profiling_info_arr[i].real_time = infos[i].real_time.count();
            profiling_info_arr[i].cpu_time = infos[i].cpu_time.count();

            profiling_info_arr[i].node_name = str_to_char_array(infos[i].node_name);
            profiling_info_arr[i].exec_type = str_to_char_array(infos[i].exec_type);
            profiling_info_arr[i].node_type = str_to_char_array(infos[i].node_type);
        }
        profiling_infos->profiling_infos = profiling_info_arr;
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_profiling_info_list_free(ov_profiling_info_list_t *profiling_infos) {
    if (!profiling_infos) {
        return;
    }
    for (int i = 0; i < profiling_infos->num; i++) {
        delete[] profiling_infos->profiling_infos[i].node_name;
        delete[] profiling_infos->profiling_infos[i].exec_type;
        delete[] profiling_infos->profiling_infos[i].node_type;
    }
    delete[] profiling_infos->profiling_infos;
    profiling_infos->profiling_infos = nullptr;
    profiling_infos->num = 0;
}

ov_status_e ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t **tensor) {
    if (!tensor || !shape.dims || element_type_map.find(type) == element_type_map.end()) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *tensor = new ov_tensor_t;
        auto tmp_type = GET_OV_ELEMENT_TYPE(type);
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.ranks, std::back_inserter(tmp_shape));
        (*tensor)->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_create_from_host_ptr(const ov_element_type_e type, const ov_shape_t shape, void *host_ptr,
                                      ov_tensor_t **tensor) {
    if (!tensor || !host_ptr || !shape.dims || element_type_map.find(type) == element_type_map.end()) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *tensor = new ov_tensor_t;
        auto tmp_type = GET_OV_ELEMENT_TYPE(type);
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.ranks, std::back_inserter(tmp_shape));
        (*tensor)->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape, host_ptr);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape) {
    if (!tensor || !shape.dims) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.ranks, std::back_inserter(tmp_shape));
        tensor->object->set_shape(tmp_shape);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape) {
    if (!tensor || !shape->dims) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto tmp_shape = tensor->object->get_shape();
        if (tmp_shape.size() > MAX_DIMENSION) {
            return ov_status_e::GENERAL_ERROR;
        }
        shape->ranks = tmp_shape.size();
        std::copy_n(tmp_shape.begin(), tmp_shape.size(), shape->dims);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type) {
    if (!tensor || !type) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto tmp_type = tensor->object->get_element_type();
        *type = GET_CAPI_ELEMENT_TYPE(tmp_type);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size) {
    if (!tensor || !elements_size) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *elements_size = tensor->object->get_size();
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size) {
    if (!tensor || !byte_size) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *byte_size = tensor->object->get_byte_size();
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_data(const ov_tensor_t* tensor, void** data) {
    if (!tensor || !data) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *data = tensor->object->data();
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_tensor_free(ov_tensor_t* tensor) {
    delete tensor;
}