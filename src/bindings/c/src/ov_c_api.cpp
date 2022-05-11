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
#include <istream>

#include "c_api/ov_c_api.h"
#include "openvino/openvino.hpp"

struct ov_core {
    std::shared_ptr<ov::Core> object;
};

struct ov_node {
    std::shared_ptr<ov::Node> object;
};

struct ov_output_node {
    std::shared_ptr<ov::Output<ov::Node>> object;
};

struct ov_model {
    std::shared_ptr<ov::Model> object;
};

struct ov_preprocess {
    std::shared_ptr<ov::preprocess::PrePostProcessor> object;
};

struct ov_preprocess_input_info {
    std::shared_ptr<ov::preprocess::InputInfo> object;
};

struct ov_preprocess_input_tensor_info {
    std::shared_ptr<ov::preprocess::InputTensorInfo> object;
};

struct ov_preprocess_output_info {
    std::shared_ptr<ov::preprocess::OutputInfo> object;
};

struct ov_preprocess_output_tensor_info {
    std::shared_ptr<ov::preprocess::OutputTensorInfo> object;
};

struct ov_preprocess_input_model_info {
    std::shared_ptr<ov::preprocess::InputModelInfo> object;
};

struct ov_preprocess_input_process_steps {
    std::shared_ptr<ov::preprocess::PreProcessSteps> object;
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

std::unordered_map <ov_element_type_e, ov::element::Type_t> g_element_type_map = {
        {ov_element_type_e::UNDEFINED, ov::element::Type_t::undefined},
        {ov_element_type_e::DYNAMIC,   ov::element::Type_t::dynamic},
        {ov_element_type_e::BOOLEAN,   ov::element::Type_t::boolean},
        {ov_element_type_e::BF16,      ov::element::Type_t::bf16},
        {ov_element_type_e::F16,       ov::element::Type_t::f16},
        {ov_element_type_e::F32,       ov::element::Type_t::f32},
        {ov_element_type_e::F64,       ov::element::Type_t::f64},
        {ov_element_type_e::I4,        ov::element::Type_t::i4},
        {ov_element_type_e::I8,        ov::element::Type_t::i8},
        {ov_element_type_e::I16,       ov::element::Type_t::i16},
        {ov_element_type_e::I32,       ov::element::Type_t::i32},
        {ov_element_type_e::I64,       ov::element::Type_t::i64},
        {ov_element_type_e::U1,        ov::element::Type_t::u1},
        {ov_element_type_e::U4,        ov::element::Type_t::u4},
        {ov_element_type_e::U8,        ov::element::Type_t::u8},
        {ov_element_type_e::U16,       ov::element::Type_t::u16},
        {ov_element_type_e::U32,       ov::element::Type_t::u32},
        {ov_element_type_e::U64,       ov::element::Type_t::u64}
};

ov_status_e ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t **tensor) {
    if (!tensor || !shape || g_element_type_map.find(type) == g_element_type_map.end()) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *tensor = new ov_tensor_t;
        auto tmp_type = g_element_type_map[type];
        ov::Shape tmp_shape;
        std::copy_if(shape, shape + 4,
                     tmp_shape.begin(),
                     [](size_t x) { return x != 0; });
        (*tensor)->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_create_from_host_ptr(const ov_element_type_e type, const ov_shape_t shape, void *host_ptr,
                                      ov_tensor_t **tensor) {
    if (!tensor || !host_ptr || !shape || g_element_type_map.find(type) == g_element_type_map.end()) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        *tensor = new ov_tensor_t;
        auto tmp_type = g_element_type_map[type];
        ov::Shape tmp_shape;
        std::copy_if(shape, shape + 4,
                     tmp_shape.begin(),
                     [](size_t x) { return x != 0; });
        (*tensor)->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape, host_ptr);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape) {
    if (!tensor || !shape) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        ov::Shape tmp_shape;
        std::copy_if(shape, shape + 4,
                     tmp_shape.begin(),
                     [](size_t x) { return x != 0;});
        tensor->object->set_shape(tmp_shape);
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape) {
    if (!tensor || !shape) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto tmp_shape = tensor->object->get_shape();
        std::copy_if(tmp_shape.begin(), tmp_shape.end(),
                     *shape,
                     [](size_t x) { return x != 0;});
    } CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type) {
    if (!tensor || !type) {
        return ov_status_e::GENERAL_ERROR;
    }
    try {
        auto tmp_type = tensor->object->get_element_type();
        for (auto it : g_element_type_map) {
            if (it.second == tmp_type) {
                *type = it.first;
                break;
            }
        }
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

void ov_tensor_free(const ov_tensor_t* tensor) {
    delete tensor;
}
