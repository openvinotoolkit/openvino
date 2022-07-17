// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "c_api/ov_c_api.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <streambuf>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/openvino.hpp"

struct ov_core {
    std::shared_ptr<ov::Core> object;
};

struct ov_output_const_node {
    std::shared_ptr<ov::Output<const ov::Node>> object;
};

struct ov_output_node {
    std::shared_ptr<ov::Output<ov::Node>> object;
};

struct ov_model {
    std::shared_ptr<ov::Model> object;
};

struct ov_preprocess_prepostprocessor {
    std::shared_ptr<ov::preprocess::PrePostProcessor> object;
};

struct ov_preprocess_inputinfo {
    ov::preprocess::InputInfo* object;
};

struct ov_preprocess_inputtensorinfo {
    ov::preprocess::InputTensorInfo* object;
};

struct ov_preprocess_outputinfo {
    ov::preprocess::OutputInfo* object;
};

struct ov_preprocess_outputtensorinfo {
    ov::preprocess::OutputTensorInfo* object;
};

struct ov_preprocess_inputmodelinfo {
    ov::preprocess::InputModelInfo* object;
};

struct ov_preprocess_preprocesssteps {
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

struct ov_layout {
    ov::Layout object;
};

struct ov_rank {
    ov::Dimension object;
};

struct ov_dimensions {
    std::vector<ov::Dimension> object;
};

struct ov_partial_shape {
    ov::Dimension rank;               // Support static rank and dynamic rank
    std::vector<ov::Dimension> dims;  // Dimemsion vector
};

struct ov_property {
    ov::AnyMap object;
};

/**
 * @variable global value for error info
 */
char const* error_infos[] = {
    "no error",
    "general error",
    "not implement",
    "network load failed",
    "input parameter mismatch",
    "cannot find the value",
    "out of bounds",
    "calloc failure",
    "invalid parameters",
    "run with unexpected error",
    "request is busy",
    "result is not ready",
    "not allocated",
    "inference start with error",
    "network is not ready",
    "inference is canceled",
    "unknown error",
};

/**
 * @struct mem_stringbuf
 * @brief This struct puts memory buffer to stringbuf.
 */
struct mem_stringbuf : std::streambuf {
    mem_stringbuf(const char* buffer, size_t sz) {
        char* bptr(const_cast<char*>(buffer));
        setg(bptr, bptr, bptr + sz);
    }

    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override {
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
struct mem_istream : virtual mem_stringbuf, std::istream {
    mem_istream(const char* buffer, size_t sz)
        : mem_stringbuf(buffer, sz),
          std::istream(static_cast<std::streambuf*>(this)) {}
};

std::map<ov_performance_mode_e, ov::hint::PerformanceMode> performance_mode_map = {
    {ov_performance_mode_e::UNDEFINED_MODE, ov::hint::PerformanceMode::UNDEFINED},
    {ov_performance_mode_e::THROUGHPUT, ov::hint::PerformanceMode::THROUGHPUT},
    {ov_performance_mode_e::LATENCY, ov::hint::PerformanceMode::LATENCY},
    {ov_performance_mode_e::CUMULATIVE_THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT}};

std::map<ov_preprocess_resizealgorithm_e, ov::preprocess::ResizeAlgorithm> resize_algorithm_map = {
    {ov_preprocess_resizealgorithm_e::RESIZE_CUBIC, ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC},
    {ov_preprocess_resizealgorithm_e::RESIZE_LINEAR, ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR},
    {ov_preprocess_resizealgorithm_e::RESIZE_NEAREST, ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST}};

std::map<ov_color_format_e, ov::preprocess::ColorFormat> color_format_map = {
    {ov_color_format_e::UNDEFINE, ov::preprocess::ColorFormat::UNDEFINED},
    {ov_color_format_e::NV12_SINGLE_PLANE, ov::preprocess::ColorFormat::NV12_SINGLE_PLANE},
    {ov_color_format_e::NV12_TWO_PLANES, ov::preprocess::ColorFormat::NV12_TWO_PLANES},
    {ov_color_format_e::I420_SINGLE_PLANE, ov::preprocess::ColorFormat::I420_SINGLE_PLANE},
    {ov_color_format_e::I420_THREE_PLANES, ov::preprocess::ColorFormat::I420_THREE_PLANES},
    {ov_color_format_e::RGB, ov::preprocess::ColorFormat::RGB},
    {ov_color_format_e::BGR, ov::preprocess::ColorFormat::BGR},
    {ov_color_format_e::RGBX, ov::preprocess::ColorFormat::RGBX},
    {ov_color_format_e::BGRX, ov::preprocess::ColorFormat::BGRX}};

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

#define GET_OV_ELEMENT_TYPE(a)   element_type_map[a]
#define GET_CAPI_ELEMENT_TYPE(a) find_ov_element_type_e(a)

#define GET_OV_COLOR_FARMAT(a) \
    (color_format_map.find(a) == color_format_map.end() ? ov::preprocess::ColorFormat::UNDEFINED : color_format_map[a])

#define CATCH_OV_EXCEPTION(StatusCode, ExceptionType) \
    catch (const InferenceEngine::ExceptionType&) {   \
        return ov_status_e::StatusCode;               \
    }

#define CATCH_OV_EXCEPTIONS                                   \
    CATCH_OV_EXCEPTION(GENERAL_ERROR, GeneralError)           \
    CATCH_OV_EXCEPTION(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_OV_EXCEPTION(NETWORK_NOT_LOADED, NetworkNotLoaded)  \
    CATCH_OV_EXCEPTION(PARAMETER_MISMATCH, ParameterMismatch) \
    CATCH_OV_EXCEPTION(NOT_FOUND, NotFound)                   \
    CATCH_OV_EXCEPTION(OUT_OF_BOUNDS, OutOfBounds)            \
    CATCH_OV_EXCEPTION(UNEXPECTED, Unexpected)                \
    CATCH_OV_EXCEPTION(REQUEST_BUSY, RequestBusy)             \
    CATCH_OV_EXCEPTION(RESULT_NOT_READY, ResultNotReady)      \
    CATCH_OV_EXCEPTION(NOT_ALLOCATED, NotAllocated)           \
    CATCH_OV_EXCEPTION(INFER_NOT_STARTED, InferNotStarted)    \
    CATCH_OV_EXCEPTION(NETWORK_NOT_READ, NetworkNotRead)      \
    CATCH_OV_EXCEPTION(INFER_CANCELLED, InferCancelled)       \
    catch (...) {                                             \
        return ov_status_e::UNEXPECTED;                       \
    }

char* str_to_char_array(const std::string& str) {
    std::unique_ptr<char> _char_array(new char[str.length() + 1]);
    char* char_array = _char_array.release();
    std::copy_n(str.begin(), str.length() + 1, char_array);
    return char_array;
}

const char* ov_get_error_info(ov_status_e status) {
    if (status > ov_status_e::UNKNOWN_ERROR)
        return error_infos[ov_status_e::UNKNOWN_ERROR];
    return error_infos[status];
}

ov_status_e ov_rank_create(ov_rank_t** rank, int64_t min_dimension, int64_t max_dimension) {
    if (!rank || min_dimension < -1 || max_dimension < -1) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_rank_t> _rank(new ov_rank_t);
        if (min_dimension != max_dimension) {
            _rank->object = ov::Dimension(min_dimension, max_dimension);
        } else {
            if (min_dimension > -1) {
                _rank->object = ov::Dimension(min_dimension);
            } else {
                _rank->object = ov::Dimension();
            }
        }
        *rank = _rank.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_rank_free(ov_rank_t* rank) {
    if (rank)
        delete rank;
}

ov_status_e ov_dimensions_create(ov_dimensions_t** dimensions) {
    if (!dimensions) {
        return ov_status_e::INVALID_PARAM;
    }
    *dimensions = nullptr;
    try {
        std::unique_ptr<ov_dimensions_t> dims(new ov_dimensions_t);
        if (!dims) {
            return ov_status_e::CALLOC_FAILED;
        }
        *dimensions = dims.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_dimensions_add(ov_dimensions_t* dimensions, int64_t min_dimension, int64_t max_dimension) {
    if (!dimensions || min_dimension < -1 || max_dimension < -1) {
        return ov_status_e::INVALID_PARAM;
    }
    dimensions->object.emplace_back(min_dimension, max_dimension);
    return ov_status_e::OK;
}

void ov_dimensions_free(ov_dimensions_t* dimensions) {
    if (dimensions)
        delete dimensions;
}

ov_status_e ov_partial_shape_create(ov_partial_shape_t** partial_shape_obj, ov_rank_t* rank, ov_dimensions_t* dims) {
    if (!partial_shape_obj || !rank) {
        return ov_status_e::INVALID_PARAM;
    }
    *partial_shape_obj = nullptr;
    try {
        std::unique_ptr<ov_partial_shape_t> partial_shape(new ov_partial_shape_t);
        if (rank->object.is_dynamic()) {
            partial_shape->rank = rank->object;
        } else {
            if (rank->object.get_length() != dims->object.size()) {
                return ov_status_e::INVALID_PARAM;
            }
            partial_shape->rank = rank->object;
            partial_shape->dims = dims->object;
        }
        *partial_shape_obj = partial_shape.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_partial_shape_free(ov_partial_shape_t* partial_shape) {
    if (partial_shape)
        delete partial_shape;
}

const char* ov_partial_shape_to_string(ov_partial_shape_t* partial_shape) {
    if (!partial_shape) {
        return str_to_char_array("Error: null partial_shape!");
    }

    // dynamic rank
    if (partial_shape->rank.is_dynamic()) {
        return str_to_char_array("?");
    }

    // static rank
    auto rank = partial_shape->rank.get_length();
    if (rank != partial_shape->dims.size()) {
        return str_to_char_array("rank error");
    }
    std::string str = std::string("{");
    int i = 0;
    for (auto& item : partial_shape->dims) {
        std::ostringstream out;
        out.str("");
        out << item;
        str += out.str();
        if (i++ < rank - 1)
            str += ",";
    }
    str += std::string("}");
    const char* res = str_to_char_array(str);

    return res;
}

ov_status_e ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape) {
    if (!partial_shape || !shape) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        if (partial_shape->rank.is_dynamic()) {
            return ov_status_e::PARAMETER_MISMATCH;
        }
        auto rank = partial_shape->rank.get_length();
        if (rank > MAX_DIMENSION) {
            return ov_status_e::PARAMETER_MISMATCH;
        }
        for (auto i = 0; i < rank; ++i) {
            auto& ov_dim = partial_shape->dims[i];
            if (ov_dim.is_static())
                shape->dims[i] = ov_dim.get_length();
            else
                return ov_status_e::PARAMETER_MISMATCH;
        }
        shape->rank = rank;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_shape_to_partial_shape(ov_shape_t* shape, ov_partial_shape_t** partial_shape) {
    if (!partial_shape || !shape) {
        return ov_status_e::INVALID_PARAM;
    }
    if (shape->rank > MAX_DIMENSION) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        *partial_shape = new ov_partial_shape_t;
        (*partial_shape)->rank = ov::Dimension(shape->rank);
        for (int i = 0; i < shape->rank; i++) {
            (*partial_shape)->dims.emplace_back(shape->dims[i]);
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_layout_create(ov_layout_t** layout, const char* layout_desc) {
    if (!layout || !layout_desc) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_layout_t> _layout(new ov_layout_t);
        _layout->object = ov::Layout(layout_desc);
        *layout = _layout.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_layout_free(ov_layout_t* layout) {
    if (layout)
        delete layout;
}

const char* ov_layout_to_string(ov_layout_t* layout) {
    if (!layout) {
        return str_to_char_array("Error: null layout!");
    }

    auto str = layout->object.to_string();
    const char* res = str_to_char_array(str);
    return res;
}

ov_status_e ov_property_create(ov_property_t** property) {
    if (!property) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_property_t> _property(new ov_property_t);
        *property = _property.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_property_free(ov_property_t* property) {
    if (property)
        delete property;
}

ov_status_e ov_property_put(ov_property_t* property, ov_property_key_e key, ov_property_value_t value) {
    if (!property || !value || key >= ov_property_key_e::MAX_KEY_VALUE) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        switch (key) {
        case ov_property_key_e::PERFORMANCE_HINT_NUM_REQUESTS: {
            uint32_t v = *(static_cast<uint32_t*>(value));
            property->object.emplace(ov::hint::num_requests(v));
            break;
        }
        case ov_property_key_e::NUM_STREAMS: {
            uint32_t v = *(static_cast<uint32_t*>(value));
            property->object.emplace(ov::num_streams(v));
            break;
        }
        case ov_property_key_e::PERFORMANCE_HINT: {
            ov_performance_mode_e m = *(static_cast<ov_performance_mode_e*>(value));
            if (m > ov_performance_mode_e::CUMULATIVE_THROUGHPUT) {
                return ov_status_e::INVALID_PARAM;
            }
            auto v = performance_mode_map[m];
            property->object.emplace(ov::hint::performance_mode(v));
            break;
        }
        case ov_property_key_e::AFFINITY: {
            ov_affinity_e v = *(static_cast<ov_affinity_e*>(value));
            if (v < ov_affinity_e::NONE || v > ov_affinity_e::HYBRID_AWARE) {
                return ov_status_e::INVALID_PARAM;
            }
            ov::Affinity affinity = static_cast<ov::Affinity>(v);
            property->object.emplace(ov::affinity(affinity));
            break;
        }
        case ov_property_key_e::INFERENCE_NUM_THREADS: {
            int32_t v = *(static_cast<int32_t*>(value));
            property->object.emplace(ov::inference_num_threads(v));
            break;
        }
        case ov_property_key_e::INFERENCE_PRECISION_HINT: {
            ov_element_type_e v = *(static_cast<ov_element_type_e*>(value));
            if (v >= ov_element_type_e::MAX) {
                return ov_status_e::INVALID_PARAM;
            }

            ov::element::Type type(static_cast<ov::element::Type_t>(v));
            property->object.emplace(ov::hint::inference_precision(type));
            break;
        }
        case ov_property_key_e::CACHE_DIR: {
            char* dir = static_cast<char*>(value);
            property->object.emplace(ov::cache_dir(std::string(dir)));
            break;
        }
        default:
            break;
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_property_value_free(ov_property_value_t value) {
    if (value) {
        char* temp = static_cast<char*>(value);
        delete temp;
    }
}

ov_status_e ov_get_openvino_version(ov_version_t* version) {
    if (!version) {
        return ov_status_e::INVALID_PARAM;
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

ov_status_e ov_core_create(const char* xml_config_file, ov_core_t** core) {
    if (!core || !xml_config_file) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_core_t> _core(new ov_core_t);
        _core->object = std::make_shared<ov::Core>(xml_config_file);
        *core = _core.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
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
        return ov_status_e::INVALID_PARAM;
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
    if (!core || !model_str || !weights || !model) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        *model = new ov_model_t;
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        _model->object = core->object->read_model(model_str, *(weights->object));
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_model_free(ov_model_t* model) {
    if (model)
        delete model;
}

ov_status_e ov_core_compile_model(const ov_core_t* core,
                                  const ov_model_t* model,
                                  const char* device_name,
                                  ov_compiled_model_t** compiled_model,
                                  const ov_property_t* property) {
    if (!core || !model || !compiled_model) {
        return ov_status_e::INVALID_PARAM;
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
        return ov_status_e::INVALID_PARAM;
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

void ov_compiled_model_free(ov_compiled_model_t* compiled_model) {
    if (compiled_model)
        delete compiled_model;
}

ov_status_e ov_core_set_property(const ov_core_t* core, const char* device_name, const ov_property_t* property) {
    if (!core || !property) {
        return ov_status_e::INVALID_PARAM;
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
        return ov_status_e::INVALID_PARAM;
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
            *value = static_cast<ov_property_value_t>(tmp);
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
            *value = static_cast<ov_property_value_t>(tmp);
            break;
        }
        case ov_property_key_e::OPTIMAL_NUMBER_OF_INFER_REQUESTS: {
            auto optimal_number_of_infer_requests =
                core->object->get_property(device_name, ov::optimal_number_of_infer_requests);
            uint32_t* temp = new uint32_t;
            *temp = optimal_number_of_infer_requests;
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::RANGE_FOR_ASYNC_INFER_REQUESTS: {
            auto range = core->object->get_property(device_name, ov::range_for_async_infer_requests);
            uint32_t* temp = new uint32_t[3];
            temp[0] = std::get<0>(range);
            temp[1] = std::get<1>(range);
            temp[2] = std::get<2>(range);
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::RANGE_FOR_STREAMS: {
            auto range = core->object->get_property(device_name, ov::range_for_streams);
            uint32_t* temp = new uint32_t[2];
            temp[0] = std::get<0>(range);
            temp[1] = std::get<1>(range);
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::FULL_DEVICE_NAME: {
            auto name = core->object->get_property(device_name, ov::device::full_name);
            char* tmp = new char[name.length() + 1];
            std::copy_n(name.begin(), name.length() + 1, tmp);
            *value = static_cast<ov_property_value_t>(tmp);
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
            *value = static_cast<ov_property_value_t>(tmp);
            break;
        }
        case ov_property_key_e::CACHE_DIR: {
            auto dir = core->object->get_property(device_name, ov::cache_dir);
            char* tmp = new char[dir.length() + 1];
            std::copy_n(dir.begin(), dir.length() + 1, tmp);
            *value = static_cast<ov_property_value_t>(tmp);
            break;
        }
        case ov_property_key_e::NUM_STREAMS: {
            auto num = core->object->get_property(device_name, ov::num_streams);
            int32_t* temp = new int32_t;
            *temp = num.num;
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::AFFINITY: {
            auto affinity = core->object->get_property(device_name, ov::affinity);
            ov_affinity_e* temp = new ov_affinity_e;
            *temp = static_cast<ov_affinity_e>(affinity);
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::INFERENCE_NUM_THREADS: {
            auto num = core->object->get_property(device_name, ov::inference_num_threads);
            int32_t* temp = new int32_t;
            *temp = num;
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::PERFORMANCE_HINT: {
            auto perf_mode = core->object->get_property(device_name, ov::hint::performance_mode);
            ov_performance_mode_e* temp = new ov_performance_mode_e;
            *temp = static_cast<ov_performance_mode_e>(perf_mode);
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::NETWORK_NAME: {
            auto name = core->object->get_property(device_name, ov::model_name);
            char* tmp = new char[name.length() + 1];
            std::copy_n(name.begin(), name.length() + 1, tmp);
            *value = static_cast<ov_property_value_t>(tmp);
            break;
        }
        case ov_property_key_e::INFERENCE_PRECISION_HINT: {
            auto infer_precision = core->object->get_property(device_name, ov::hint::inference_precision);
            ov_element_type_e* temp = new ov_element_type_e;
            *temp = static_cast<ov_element_type_e>(ov::element::Type_t(infer_precision));
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::OPTIMAL_BATCH_SIZE: {
            auto batch_size = core->object->get_property(device_name, ov::optimal_batch_size);
            uint32_t* temp = new uint32_t;
            *temp = batch_size;
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::MAX_BATCH_SIZE: {
            auto batch_size = core->object->get_property(device_name, ov::max_batch_size);
            uint32_t* temp = new uint32_t;
            *temp = batch_size;
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        case ov_property_key_e::PERFORMANCE_HINT_NUM_REQUESTS: {
            auto num_requests = core->object->get_property(device_name, ov::hint::num_requests);
            uint32_t* temp = new uint32_t;
            *temp = num_requests;
            *value = static_cast<ov_property_value_t>(temp);
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
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto available_devices = core->object->get_available_devices();
        devices->num_devices = available_devices.size();
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
    for (int i = 0; i < devices->num_devices; i++) {
        if (devices->devices[i]) {
            delete[] devices->devices[i];
        }
    }
    if (devices->devices)
        delete[] devices->devices;
    devices->devices = nullptr;
    devices->num_devices = 0;
}

ov_status_e ov_core_import_model(const ov_core_t* core,
                                 const char* content,
                                 const size_t content_size,
                                 const char* device_name,
                                 ov_compiled_model_t** compiled_model) {
    if (!core || !content || !device_name || !compiled_model) {
        return ov_status_e::INVALID_PARAM;
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
        return ov_status_e::INVALID_PARAM;
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
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_core_versions_free(ov_core_version_list_t* versions) {
    if (!versions) {
        return;
    }
    for (int i = 0; i < versions->num_vers; i++) {
        if (versions->versions[i].device_name)
            delete[] versions->versions[i].device_name;
        if (versions->versions[i].buildNumber)
            delete[] versions->versions[i].buildNumber;
        if (versions->versions[i].description)
            delete[] versions->versions[i].description;
    }
    if (versions->versions)
        delete[] versions->versions;
    versions->versions = nullptr;
}

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

void ov_free(const char* content) {
    if (content)
        delete content;
}

ov_status_e ov_preprocess_prepostprocessor_create(const ov_model_t* model,
                                                  ov_preprocess_prepostprocessor_t** preprocess) {
    if (!model || !preprocess) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_prepostprocessor_t> _preprocess(new ov_preprocess_prepostprocessor_t);
        _preprocess->object = std::make_shared<ov::preprocess::PrePostProcessor>(model->object);
        *preprocess = _preprocess.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_prepostprocessor_free(ov_preprocess_prepostprocessor_t* preprocess) {
    if (preprocess)
        delete preprocess;
}

ov_status_e ov_preprocess_prepostprocessor_input(const ov_preprocess_prepostprocessor_t* preprocess,
                                                 ov_preprocess_inputinfo_t** preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputinfo_t> _preprocess_input_info(new ov_preprocess_inputinfo_t);
        _preprocess_input_info->object = &(preprocess->object->input());
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_input_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                         const char* tensor_name,
                                                         ov_preprocess_inputinfo_t** preprocess_input_info) {
    if (!preprocess || !tensor_name || !preprocess_input_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputinfo_t> _preprocess_input_info(new ov_preprocess_inputinfo_t);
        _preprocess_input_info->object = &(preprocess->object->input(tensor_name));
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_input_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                          const size_t tensor_index,
                                                          ov_preprocess_inputinfo_t** preprocess_input_info) {
    if (!preprocess || !preprocess_input_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputinfo_t> _preprocess_input_info(new ov_preprocess_inputinfo_t);
        _preprocess_input_info->object = &(preprocess->object->input(tensor_index));
        *preprocess_input_info = _preprocess_input_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_inputinfo_free(ov_preprocess_inputinfo_t* preprocess_input_info) {
    if (preprocess_input_info)
        delete preprocess_input_info;
}

ov_status_e ov_preprocess_inputinfo_tensor(const ov_preprocess_inputinfo_t* preprocess_input_info,
                                           ov_preprocess_inputtensorinfo_t** preprocess_input_tensor_info) {
    if (!preprocess_input_info || !preprocess_input_tensor_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputtensorinfo_t> _preprocess_input_tensor_info(
            new ov_preprocess_inputtensorinfo_t);
        _preprocess_input_tensor_info->object = &(preprocess_input_info->object->tensor());
        *preprocess_input_tensor_info = _preprocess_input_tensor_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_inputtensorinfo_free(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info) {
    if (preprocess_input_tensor_info)
        delete preprocess_input_tensor_info;
}

ov_status_e ov_preprocess_inputinfo_preprocess(const ov_preprocess_inputinfo_t* preprocess_input_info,
                                               ov_preprocess_preprocesssteps_t** preprocess_input_steps) {
    if (!preprocess_input_info || !preprocess_input_steps) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_preprocesssteps_t> _preprocess_input_steps(new ov_preprocess_preprocesssteps_t);
        _preprocess_input_steps->object = &(preprocess_input_info->object->preprocess());
        *preprocess_input_steps = _preprocess_input_steps.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_preprocesssteps_free(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps) {
    if (preprocess_input_process_steps)
        delete preprocess_input_process_steps;
}

ov_status_e ov_preprocess_preprocesssteps_resize(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                                 const ov_preprocess_resizealgorithm_e resize_algorithm) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_process_steps->object->resize(resize_algorithm_map[resize_algorithm]);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_element_type(
    ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
    const ov_element_type_e element_type) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_element_type(GET_OV_ELEMENT_TYPE(element_type));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_from(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                                   const ov_tensor_t* tensor) {
    if (!preprocess_input_tensor_info || !tensor) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_from(*(tensor->object));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_layout(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                                     ov_layout_t* layout) {
    if (!preprocess_input_tensor_info || !layout) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_layout(layout->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_color_format(
    ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
    const ov_color_format_e colorFormat) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_color_format(GET_OV_COLOR_FARMAT(colorFormat));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputtensorinfo_set_spatial_static_shape(
    ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
    const size_t input_height,
    const size_t input_width) {
    if (!preprocess_input_tensor_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_tensor_info->object->set_spatial_static_shape(input_height, input_width);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocesssteps_convert_element_type(
    ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
    const ov_element_type_e element_type) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_process_steps->object->convert_element_type(GET_OV_ELEMENT_TYPE(element_type));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_preprocesssteps_convert_color(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                                        const ov_color_format_e colorFormat) {
    if (!preprocess_input_process_steps) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_process_steps->object->convert_color(GET_OV_COLOR_FARMAT(colorFormat));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_output(const ov_preprocess_prepostprocessor_t* preprocess,
                                                  ov_preprocess_outputinfo_t** preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        *preprocess_output_info = new ov_preprocess_outputinfo_t;
        (*preprocess_output_info)->object = &(preprocess->object->output());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_output_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                           const size_t tensor_index,
                                                           ov_preprocess_outputinfo_t** preprocess_output_info) {
    if (!preprocess || !preprocess_output_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputinfo_t> _preprocess_output_info(new ov_preprocess_outputinfo_t);
        _preprocess_output_info->object = &(preprocess->object->output(tensor_index));
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_output_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                          const char* tensor_name,
                                                          ov_preprocess_outputinfo_t** preprocess_output_info) {
    if (!preprocess || !tensor_name || !preprocess_output_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputinfo_t> _preprocess_output_info(new ov_preprocess_outputinfo_t);
        _preprocess_output_info->object = &(preprocess->object->output(tensor_name));
        *preprocess_output_info = _preprocess_output_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_outputinfo_free(ov_preprocess_outputinfo_t* preprocess_output_info) {
    if (preprocess_output_info)
        delete preprocess_output_info;
}

ov_status_e ov_preprocess_outputinfo_tensor(ov_preprocess_outputinfo_t* preprocess_output_info,
                                            ov_preprocess_outputtensorinfo_t** preprocess_output_tensor_info) {
    if (!preprocess_output_info || !preprocess_output_tensor_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_outputtensorinfo_t> _preprocess_output_tensor_info(
            new ov_preprocess_outputtensorinfo_t);
        _preprocess_output_tensor_info->object = &(preprocess_output_info->object->tensor());
        *preprocess_output_tensor_info = _preprocess_output_tensor_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_outputtensorinfo_free(ov_preprocess_outputtensorinfo_t* preprocess_output_tensor_info) {
    if (preprocess_output_tensor_info)
        delete preprocess_output_tensor_info;
}

ov_status_e ov_preprocess_output_set_element_type(ov_preprocess_outputtensorinfo_t* preprocess_output_tensor_info,
                                                  const ov_element_type_e element_type) {
    if (!preprocess_output_tensor_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_output_tensor_info->object->set_element_type(GET_OV_ELEMENT_TYPE(element_type));
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_inputinfo_model(ov_preprocess_inputinfo_t* preprocess_input_info,
                                          ov_preprocess_inputmodelinfo_t** preprocess_input_model_info) {
    if (!preprocess_input_info || !preprocess_input_model_info) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_preprocess_inputmodelinfo_t> _preprocess_input_model_info(
            new ov_preprocess_inputmodelinfo_t);
        _preprocess_input_model_info->object = &(preprocess_input_info->object->model());
        *preprocess_input_model_info = _preprocess_input_model_info.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_preprocess_inputmodelinfo_free(ov_preprocess_inputmodelinfo_t* preprocess_input_model_info) {
    if (preprocess_input_model_info)
        delete preprocess_input_model_info;
}

ov_status_e ov_preprocess_inputmodelinfo_set_layout(ov_preprocess_inputmodelinfo_t* preprocess_input_model_info,
                                                    ov_layout_t* layout) {
    if (!preprocess_input_model_info || !layout) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        preprocess_input_model_info->object->set_layout(layout->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_preprocess_prepostprocessor_build(const ov_preprocess_prepostprocessor_t* preprocess,
                                                 ov_model_t** model) {
    if (!preprocess || !model) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        _model->object = preprocess->object->build();
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model, ov_model_t** model) {
    if (!compiled_model || !model) {
        return ov_status_e::INVALID_PARAM;
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

ov_status_e ov_compiled_model_inputs(const ov_compiled_model_t* compiled_model, ov_output_node_list_t* input_nodes) {
    if (!compiled_model || !input_nodes) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto inputs = compiled_model->object->inputs();
        int num = inputs.size();
        input_nodes->num = num;
        input_nodes->output_nodes = new ov_output_const_node_t[num];
        for (int i = 0; i < num; i++) {
            input_nodes->output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(inputs[i]));
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_outputs(const ov_compiled_model_t* compiled_model, ov_output_node_list_t* output_nodes) {
    if (!compiled_model || !output_nodes) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto outputs = compiled_model->object->outputs();
        int num = outputs.size();
        output_nodes->num = num;
        output_nodes->output_nodes = new ov_output_const_node_t[num];
        for (int i = 0; i < num; i++) {
            output_nodes->output_nodes[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(outputs[i]));
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model,
                                                   ov_infer_request_t** infer_request) {
    if (!compiled_model || !infer_request) {
        return ov_status_e::INVALID_PARAM;
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

ov_status_e ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, const ov_property_t* property) {
    if (!compiled_model || !property) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        compiled_model->object->set_property(property->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                                           const ov_property_key_e key,
                                           ov_property_value_t* value) {
    if (!compiled_model || !value) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        switch (key) {
        case ov_property_key_e::SUPPORTED_PROPERTIES: {
            auto supported_properties = compiled_model->object->get_property(ov::supported_properties);
            std::string tmp_s;
            for (const auto& i : supported_properties) {
                tmp_s = tmp_s + "\n" + i;
            }
            char* temp = new char[tmp_s.length() + 1];
            std::copy_n(tmp_s.c_str(), tmp_s.length() + 1, temp);
            *value = static_cast<ov_property_value_t>(temp);
            break;
        }
        default:
            break;
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_compiled_model_export_model(const ov_compiled_model_t* compiled_model, const char* export_model_path) {
    if (!compiled_model || !export_model_path) {
        return ov_status_e::INVALID_PARAM;
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

void ov_infer_request_free(ov_infer_request_t* infer_request) {
    if (infer_request)
        delete infer_request;
}

ov_status_e ov_infer_request_set_tensor(ov_infer_request_t* infer_request,
                                        const char* tensor_name,
                                        const ov_tensor_t* tensor) {
    if (!infer_request || !tensor_name || !tensor) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        infer_request->object->set_tensor(tensor_name, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request,
                                              size_t idx,
                                              const ov_tensor_t* tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        infer_request->object->set_input_tensor(idx, *tensor->object);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_tensor(const ov_infer_request_t* infer_request,
                                        const char* tensor_name,
                                        ov_tensor_t** tensor) {
    if (!infer_request || !tensor_name || !tensor) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_tensor(tensor_name);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_out_tensor(const ov_infer_request_t* infer_request, size_t idx, ov_tensor_t** tensor) {
    if (!infer_request || !tensor) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        ov::Tensor tensor_get = infer_request->object->get_output_tensor(idx);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(tensor_get));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_infer(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        infer_request->object->infer();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_cancel(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        infer_request->object->cancel();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_start_async(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        infer_request->object->start_async();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_wait(ov_infer_request_t* infer_request) {
    if (!infer_request) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        infer_request->object->wait();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback) {
    if (!infer_request || !callback) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto func = [callback](std::exception_ptr ex) {
            callback->callback_func(callback->args);
        };
        infer_request->object->set_callback(func);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_infer_request_get_profiling_info(ov_infer_request_t* infer_request,
                                                ov_profiling_info_list_t* profiling_infos) {
    if (!infer_request || !profiling_infos) {
        return ov_status_e::INVALID_PARAM;
    }

    try {
        auto infos = infer_request->object->get_profiling_info();
        int num = infos.size();
        profiling_infos->num = num;
        ov_profiling_info_t* profiling_info_arr = new ov_profiling_info_t[num];
        for (int i = 0; i < num; i++) {
            profiling_info_arr[i].status = (ov_profiling_info_t::Status)infos[i].status;
            profiling_info_arr[i].real_time = infos[i].real_time.count();
            profiling_info_arr[i].cpu_time = infos[i].cpu_time.count();

            profiling_info_arr[i].node_name = str_to_char_array(infos[i].node_name);
            profiling_info_arr[i].exec_type = str_to_char_array(infos[i].exec_type);
            profiling_info_arr[i].node_type = str_to_char_array(infos[i].node_type);
        }
        profiling_infos->profiling_infos = profiling_info_arr;
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_profiling_info_list_free(ov_profiling_info_list_t* profiling_infos) {
    if (!profiling_infos) {
        return;
    }
    for (int i = 0; i < profiling_infos->num; i++) {
        if (profiling_infos->profiling_infos[i].node_name)
            delete[] profiling_infos->profiling_infos[i].node_name;
        if (profiling_infos->profiling_infos[i].exec_type)
            delete[] profiling_infos->profiling_infos[i].exec_type;
        if (profiling_infos->profiling_infos[i].node_type)
            delete[] profiling_infos->profiling_infos[i].node_type;
    }
    if (profiling_infos->profiling_infos)
        delete[] profiling_infos->profiling_infos;
    profiling_infos->profiling_infos = nullptr;
    profiling_infos->num = 0;
}

ov_status_e ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor) {
    if (!tensor || element_type_map.find(type) == element_type_map.end()) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        auto tmp_type = GET_OV_ELEMENT_TYPE(type);
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        _tensor->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape);
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_create_from_host_ptr(const ov_element_type_e type,
                                           const ov_shape_t shape,
                                           void* host_ptr,
                                           ov_tensor_t** tensor) {
    if (!tensor || !host_ptr || element_type_map.find(type) == element_type_map.end()) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        auto tmp_type = GET_OV_ELEMENT_TYPE(type);
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        _tensor->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape, host_ptr);
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape) {
    if (!tensor) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        tensor->object->set_shape(tmp_shape);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape) {
    if (!tensor) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto tmp_shape = tensor->object->get_shape();
        if (tmp_shape.size() > MAX_DIMENSION) {
            return ov_status_e::GENERAL_ERROR;
        }
        shape->rank = tmp_shape.size();
        std::copy_n(tmp_shape.begin(), tmp_shape.size(), shape->dims);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type) {
    if (!tensor || !type) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        auto tmp_type = tensor->object->get_element_type();
        *type = GET_CAPI_ELEMENT_TYPE(tmp_type);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size) {
    if (!tensor || !elements_size) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        *elements_size = tensor->object->get_size();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size) {
    if (!tensor || !byte_size) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        *byte_size = tensor->object->get_byte_size();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_data(const ov_tensor_t* tensor, void** data) {
    if (!tensor || !data) {
        return ov_status_e::INVALID_PARAM;
    }
    try {
        *data = tensor->object->data();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_tensor_free(ov_tensor_t* tensor) {
    if (tensor)
        delete tensor;
}