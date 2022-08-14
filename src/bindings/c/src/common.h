// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>
#include <iterator>
#include <map>
#include <streambuf>
#include <string>

#include "openvino/openvino.hpp"

// TODO: we need to catch ov::Exception instead of ie::Exception
#include "details/ie_exception.hpp"

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
 * @struct ov_output_const_node
 * @brief This is an interface of ov::Output<const ov::Node>
 */
struct ov_output_const_node {
    std::shared_ptr<ov::Output<const ov::Node>> object;
};

/**
 * @struct ov_output_node
 * @brief This is an interface of ov::Output<ov::Node>
 */
struct ov_output_node {
    std::shared_ptr<ov::Output<ov::Node>> object;
};

/**
 * @struct ov_property
 * @brief This is an interface of property
 */
struct ov_property {
    ov::AnyMap object;
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
 * @struct ov_rank
 * @brief This is an interface of ov::Dimension
 */
struct ov_rank {
    ov::Dimension object;
};

/**
 * @struct ov_dimension
 * @brief This is an interface of ov::Dimension
 */
struct ov_dimension {
    ov::Dimension object;
};

/**
 * @struct ov_dimensions
 * @brief This is an interface of std::vector<ov::Dimension>
 */
struct ov_dimensions {
    std::vector<ov::Dimension> object;
};

/**
 * @struct ov_partial_shape
 * @brief It represents a shape that may be partially or totally dynamic.
 * A PartialShape may have:
 * Dynamic rank. (Informal notation: `?`)
 * Static rank, but dynamic dimensions on some or all axes.
 *     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
 * Static rank, and static dimensions on all axes.
 *     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
 *
 * An interface to make user can initialize ov_partial_shape_t
 */
struct ov_partial_shape {
    ov::Dimension rank;
    std::vector<ov::Dimension> dims;
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
 * @struct ov_preprocess_inputinfo
 * @brief This is an interface of ov::preprocess::InputInfo
 */
struct ov_preprocess_inputinfo {
    ov::preprocess::InputInfo* object;
};

/**
 * @struct ov_preprocess_inputtensorinfo
 * @brief This is an interface of ov::preprocess::InputTensorInfo
 */
struct ov_preprocess_inputtensorinfo {
    ov::preprocess::InputTensorInfo* object;
};

/**
 * @struct ov_preprocess_outputinfo
 * @brief This is an interface of ov::preprocess::OutputInfo
 */
struct ov_preprocess_outputinfo {
    ov::preprocess::OutputInfo* object;
};

/**
 * @struct ov_preprocess_outputtensorinfo
 * @brief This is an interface of ov::preprocess::OutputTensorInfo
 */
struct ov_preprocess_outputtensorinfo {
    ov::preprocess::OutputTensorInfo* object;
};

/**
 * @struct ov_preprocess_inputmodelinfo
 * @brief This is an interface of ov::preprocess::InputModelInfo
 */
struct ov_preprocess_inputmodelinfo {
    ov::preprocess::InputModelInfo* object;
};

/**
 * @struct ov_preprocess_preprocesssteps
 * @brief This is an interface of ov::preprocess::PreProcessSteps
 */
struct ov_preprocess_preprocesssteps {
    ov::preprocess::PreProcessSteps* object;
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
ov_element_type_e find_ov_element_type_e(ov::element::Type type);
ov::element::Type get_element_type(ov_element_type_e type);
