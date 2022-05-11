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

// CompiledModel
OPENVINO_C_API(ov_status_e) ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model,
                                                        ov_infer_request_t **infer_request) {
    if (compiled_model == nullptr || infer_request == nullptr) {
        return ov_status_e::GENERAL_ERROR;
    }

    try {
        *infer_request = new ov_infer_request_t;
        ov::InferRequest inferReq = compiled_model->object.get()->create_infer_request();
        (*infer_request)->object = std::make_shared<ov::InferRequest>(std::move(inferReq));
    } CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

OPENVINO_C_API(void) ov_compiled_model_free(ov_compiled_model_t *compiled_model) {
    if (compiled_model) {
        delete compiled_model;
        compiled_model = NULL;
    }
}

