// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cassert>
#include <fstream>
#include <iterator>
#include <map>
#include <streambuf>
#include <string>

#include "details/ie_exception.hpp"
#include "openvino/core/except.hpp"
#include "openvino/openvino.hpp"

#define CATCH_IE_EXCEPTION(StatusCode, ExceptionType)                  \
    catch (const InferenceEngine::ExceptionType& ex) {                 \
        dup_last_err_msg(ex.what());                                   \
        return ov_status_e::StatusCode;                                \
    }

#define CATCH_OV_EXCEPTION(StatusCode, ExceptionType)                  \
    catch (const ov::ExceptionType& ex) {                              \
        dup_last_err_msg(ex.what());                                   \
        return ov_status_e::StatusCode;                                \
    }

#define CATCH_OV_EXCEPTIONS                                   \
    CATCH_OV_EXCEPTION(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_OV_EXCEPTION(GENERAL_ERROR, Exception)              \
    CATCH_IE_EXCEPTION(GENERAL_ERROR, GeneralError)           \
    CATCH_IE_EXCEPTION(NOT_IMPLEMENTED, NotImplemented)       \
    CATCH_IE_EXCEPTION(NETWORK_NOT_LOADED, NetworkNotLoaded)  \
    CATCH_IE_EXCEPTION(PARAMETER_MISMATCH, ParameterMismatch) \
    CATCH_IE_EXCEPTION(NOT_FOUND, NotFound)                   \
    CATCH_IE_EXCEPTION(OUT_OF_BOUNDS, OutOfBounds)            \
    CATCH_IE_EXCEPTION(UNEXPECTED, Unexpected)                \
    CATCH_IE_EXCEPTION(REQUEST_BUSY, RequestBusy)             \
    CATCH_IE_EXCEPTION(RESULT_NOT_READY, ResultNotReady)      \
    CATCH_IE_EXCEPTION(NOT_ALLOCATED, NotAllocated)           \
    CATCH_IE_EXCEPTION(INFER_NOT_STARTED, InferNotStarted)    \
    CATCH_IE_EXCEPTION(NETWORK_NOT_READ, NetworkNotRead)      \
    CATCH_IE_EXCEPTION(INFER_CANCELLED, InferCancelled)       \
    catch (...) {                                             \
        dup_last_err_msg("An unknown exception occurred");    \
        return ov_status_e::UNKNOW_EXCEPTION;                 \
    }

#define GET_PROPERTY_FROM_ARGS_LIST                     \
    std::string property_key = va_arg(args_ptr, char*); \
    std::string _value = va_arg(args_ptr, char*);       \
    ov::Any value = _value;                             \
    property[property_key] = value;

/**
 * @struct ov_core
 * @brief This struct represents OpenVINO Core entity.
 */
struct ov_core {
    std::shared_ptr<ov::Core> object;
};

/**
 * @struct ov_model
 * @brief This is an interface of ov::Model
 */
struct ov_model {
    std::shared_ptr<ov::Model> object;
};

/**
 * @struct ov_output_const_port
 * @brief This is an interface of ov::Output<const ov::Node>
 */
struct ov_output_const_port {
    std::shared_ptr<ov::Output<const ov::Node>> object;
};

/**
 * @struct ov_output_port
 * @brief This is an interface of ov::Output<ov::Node>
 */
struct ov_output_port {
    std::shared_ptr<ov::Output<ov::Node>> object;
};

/**
 * @struct ov_compiled_model
 * @brief This is an interface of ov::CompiledModel
 */
struct ov_compiled_model {
    std::shared_ptr<ov::CompiledModel> object;
};

/**
 * @struct ov_infer_request
 * @brief This is an interface of ov::InferRequest
 */
struct ov_infer_request {
    std::shared_ptr<ov::InferRequest> object;
};

/**
 * @struct ov_layout
 * @brief This is an interface of ov::Layout
 */
struct ov_layout {
    ov::Layout object;
};

/**
 * @struct ov_tensor
 * @brief This is an interface of ov_tensor
 */
struct ov_tensor {
    std::shared_ptr<ov::Tensor> object;
};

/**
 * @struct ov_preprocess_prepostprocessor
 * @brief This is an interface of ov::preprocess::PrePostProcessor
 */
struct ov_preprocess_prepostprocessor {
    std::shared_ptr<ov::preprocess::PrePostProcessor> object;
};

/**
 * @struct ov_preprocess_input_info
 * @brief This is an interface of ov::preprocess::InputInfo
 */
struct ov_preprocess_input_info {
    ov::preprocess::InputInfo* object;
};

/**
 * @struct ov_preprocess_input_tensor_info
 * @brief This is an interface of ov::preprocess::InputTensorInfo
 */
struct ov_preprocess_input_tensor_info {
    ov::preprocess::InputTensorInfo* object;
};

/**
 * @struct ov_preprocess_output_info
 * @brief This is an interface of ov::preprocess::OutputInfo
 */
struct ov_preprocess_output_info {
    ov::preprocess::OutputInfo* object;
};

/**
 * @struct ov_preprocess_output_tensor_info
 * @brief This is an interface of ov::preprocess::OutputTensorInfo
 */
struct ov_preprocess_output_tensor_info {
    ov::preprocess::OutputTensorInfo* object;
};

/**
 * @struct ov_preprocess_input_model_info
 * @brief This is an interface of ov::preprocess::InputModelInfo
 */
struct ov_preprocess_input_model_info {
    ov::preprocess::InputModelInfo* object;
};

/**
 * @struct ov_preprocess_preprocess_steps
 * @brief This is an interface of ov::preprocess::PreProcessSteps
 */
struct ov_preprocess_preprocess_steps {
    ov::preprocess::PreProcessSteps* object;
};

/**
 * @struct ov_remote_context
 * @brief This is an interface of ov::RemoteContext
 */
struct ov_remote_context {
    std::shared_ptr<ov::RemoteContext> object;
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

char* str_to_char_array(const std::string& str);
ov::element::Type get_element_type(ov_element_type_e type);
void dup_last_err_msg(const std::string msg);
