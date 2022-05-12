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
        preprocess_input_tensor_info->object->set_layout(layout);
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
        preprocess_input_model_info->object->set_layout(layout);
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