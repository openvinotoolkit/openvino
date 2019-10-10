// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file with common inference engine definitions.
 * @file ie_common.h
 */
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <ostream>
#include <algorithm>
#include <cstdlib>
#include <details/ie_exception.hpp>

#include "ie_unicode.hpp"

namespace InferenceEngine {
/**
 * @brief Represents tensor size.
 * The order is opposite to the order in Caffe*: (w,h,n,b) where the most frequently changing element in memory is first.
 */
using SizeVector = std::vector<size_t>;

/**
 * @brief This class represents the generic layer.
 */
class CNNLayer;

/**
 * @brief A smart pointer to the CNNLayer
 */
using CNNLayerPtr = std::shared_ptr<CNNLayer>;
/**
 * @brief A smart weak pointer to the CNNLayer
 */
using  CNNLayerWeakPtr = std::weak_ptr<CNNLayer>;

/**
 * @brief The main data representation node
 */
class Data;

/**
 * @brief Smart pointer to Data
 */
using DataPtr = std::shared_ptr<Data>;

/**
 * @brief Smart pointer to constant Data
 */
using CDataPtr = std::shared_ptr<const Data>;

/**
 * @brief Smart weak pointer to Data
 */
using DataWeakPtr = std::weak_ptr<Data>;

/**
 * @union UserValue
 * @brief The method holds the user values to enable binding of data per graph node.
 */
union UserValue {
    int v_int;
    float v_float;
    void *v_ptr;
};

/**
 * @enum Layout
 * @brief Layouts that the inference engine supports
 */
enum Layout : uint8_t {
    ANY = 0,           // "any" layout

    // I/O data layouts
    NCHW = 1,
    NHWC = 2,
    NCDHW = 3,
    NDHWC = 4,

    // weight layouts
    OIHW = 64,

    // Scalar
    SCALAR = 95,

    // bias layouts
    C = 96,

    // Single image layout (for mean image)
    CHW = 128,

    // 2D
    HW = 192,
    NC = 193,
    CN = 194,

    BLOCKED = 200,
};
inline std::ostream & operator << (std::ostream &out, const Layout & p) {
    switch (p) {
#define PRINT_LAYOUT(name)\
        case name : out << #name; break;

            PRINT_LAYOUT(ANY);
            PRINT_LAYOUT(NCHW);
            PRINT_LAYOUT(NHWC);
            PRINT_LAYOUT(NCDHW);
            PRINT_LAYOUT(NDHWC);
            PRINT_LAYOUT(OIHW);
            PRINT_LAYOUT(C);
            PRINT_LAYOUT(CHW);
            PRINT_LAYOUT(HW);
            PRINT_LAYOUT(NC);
            PRINT_LAYOUT(CN);
            PRINT_LAYOUT(BLOCKED);
#undef PRINT_LAYOUT
            default:
                 out << static_cast<int>(p);
            break;
        }
        return out;
    }

/**
 * @enum ColorFormat
 * @brief Extra information about input color format for preprocessing
 */
enum ColorFormat : uint32_t {
    RAW = 0u,    ///< Plain blob (default), no extra color processing required
    RGB,         ///< RGB color format
    BGR,         ///< BGR color format, default in DLDT
    RGBX,        ///< RGBX color format with X ignored during inference
    BGRX,        ///< BGRX color format with X ignored during inference
    NV12,        ///< NV12 color format represented as compound Y+UV blob
};
inline std::ostream & operator << (std::ostream &out, const ColorFormat & fmt) {
    switch (fmt) {
#define PRINT_COLOR_FORMAT(name) \
    case name : out << #name; break;

        PRINT_COLOR_FORMAT(RAW);
        PRINT_COLOR_FORMAT(RGB);
        PRINT_COLOR_FORMAT(BGR);
        PRINT_COLOR_FORMAT(RGBX);
        PRINT_COLOR_FORMAT(BGRX);
        PRINT_COLOR_FORMAT(NV12);

#undef PRINT_COLOR_FORMAT

        default: out << static_cast<uint32_t>(fmt); break;
    }
    return out;
}

/**
 * @struct InferenceEngineProfileInfo
 * @brief Represents basic inference profiling information per layer.
 * If the layer is executed using tiling, the sum time per each tile is indicated as the total execution time.
 * Due to parallel execution, the total execution time for all layers might be greater than the total inference time.
 */
struct InferenceEngineProfileInfo {
    /**
     * @brief Defines the general status of the layer
     */
    enum LayerStatus {
        NOT_RUN,
        OPTIMIZED_OUT,
        EXECUTED
    };

    LayerStatus status;
    /**
     * @brief The absolute time in microseconds that the layer ran (in total)
     */
    long long realTime_uSec;
    /**
     * @brief The net host cpu time that the layer ran
     */
    long long cpu_uSec;

    /**
     * @brief An execution type of unit
     */
    char exec_type[256] = {};

    /**
     * @brief A layer type
     */
    char layer_type[256] = {};

    /**
     * @brief An execution index of the unit
     */
    unsigned execution_index;
};


/**
 * @enum StatusCode
 * @brief This enum contains codes for all possible return values of the interface functions
 */
enum StatusCode : int {
    OK = 0,
    GENERAL_ERROR = -1,
    NOT_IMPLEMENTED = -2,
    NETWORK_NOT_LOADED = -3,
    PARAMETER_MISMATCH = -4,
    NOT_FOUND = -5,
    OUT_OF_BOUNDS = -6,
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    UNEXPECTED = -7,
    REQUEST_BUSY = -8,
    RESULT_NOT_READY = -9,
    NOT_ALLOCATED = -10,
    INFER_NOT_STARTED = -11,
    NETWORK_NOT_READ = -12
};

/**
 * @struct ResponseDesc
 * @brief  Represents detailed information for an error
 */
struct ResponseDesc {
    /**
     * @brief A character buffer that holds the detailed information for an error.
     */
    char msg[256] = {};
};

/** @brief This class represents StatusCode::GENERIC_ERROR exception */
class GeneralError : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::NOT_IMPLEMENTED exception */
class NotImplemented : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::NETWORK_NOT_LOADED exception */
class NetworkNotLoaded : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::PARAMETER_MISMATCH exception */
class ParameterMismatch : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::NOT_FOUND exception */
class NotFound : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::OUT_OF_BOUNDS exception */
class OutOfBounds : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::UNEXPECTED exception */
class Unexpected : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::REQUEST_BUSY exception */
class RequestBusy : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::RESULT_NOT_READY exception */
class ResultNotReady : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::NOT_ALLOCATED exception */
class NotAllocated : public std::logic_error
{ using std::logic_error::logic_error; };

/** @brief This class represents StatusCode::INFER_NOT_STARTED exception */
class InferNotStarted : public std::logic_error
{ using std::logic_error::logic_error; };
}  // namespace InferenceEngine

/** @brief This class represents StatusCode::NETWORK_NOT_READ exception */
class NetworkNotRead : public std::logic_error
{ using std::logic_error::logic_error; };

#if defined(_WIN32)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
#else
    #define __PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif
