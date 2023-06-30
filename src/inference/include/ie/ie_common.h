// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file with common inference engine definitions.
 *
 * @file ie_common.h
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ie_api.h"

IE_SUPPRESS_DEPRECATED_START
#ifndef NDEBUG
#    include <cassert>
#endif
namespace InferenceEngine {
/**
 * @brief Represents tensor size.
 *
 * The order is opposite to the order in Caffe*: (w,h,n,b) where the most frequently changing element in memory is
 * first.
 */
using SizeVector = std::vector<size_t>;

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
union INFERENCE_ENGINE_1_0_DEPRECATED UserValue {
    int v_int;      //!< An integer value
    float v_float;  //!< A floating point value
    void* v_ptr;    //!< A pointer to a void
};

/**
 * @enum Layout
 * @brief Layouts that the inference engine supports
 */
enum INFERENCE_ENGINE_1_0_DEPRECATED Layout : uint8_t {
    ANY = 0,  //!< "any" layout

    // I/O data layouts
    NCHW = 1,   //!< NCHW layout for input / output blobs
    NHWC = 2,   //!< NHWC layout for input / output blobs
    NCDHW = 3,  //!< NCDHW layout for input / output blobs
    NDHWC = 4,  //!< NDHWC layout for input / output blobs

    // weight layouts
    OIHW = 64,    //!< OIHW layout for operation weights
    GOIHW = 65,   //!< GOIHW layout for operation weights
    OIDHW = 66,   //!< OIDHW layout for operation weights
    GOIDHW = 67,  //!< GOIDHW layout for operation weights

    // Scalar
    SCALAR = 95,  //!< A scalar layout

    // bias layouts
    C = 96,  //!< A bias layout for operation

    // Single image layouts
    CHW = 128,  //!< A single image layout (e.g. for mean image)
    HWC = 129,  //!< A single image layout (e.g. for mean image)

    // 2D
    HW = 192,  //!< HW 2D layout
    NC = 193,  //!< NC 2D layout
    CN = 194,  //!< CN 2D layout

    BLOCKED = 200,  //!< A blocked layout
};

/**
 * @brief Prints a string representation of InferenceEngine::Layout to a stream
 * @param out An output stream to send to
 * @param p A layout value to print to a stream
 * @return A reference to the `out` stream
 */
INFERENCE_ENGINE_1_0_DEPRECATED inline std::ostream& operator<<(std::ostream& out, const Layout& p) {
    switch (p) {
#define PRINT_LAYOUT(name) \
    case name:             \
        out << #name;      \
        break;

        PRINT_LAYOUT(ANY);
        PRINT_LAYOUT(NCHW);
        PRINT_LAYOUT(NHWC);
        PRINT_LAYOUT(NCDHW);
        PRINT_LAYOUT(NDHWC);
        PRINT_LAYOUT(OIHW);
        PRINT_LAYOUT(GOIHW);
        PRINT_LAYOUT(OIDHW);
        PRINT_LAYOUT(GOIDHW);
        PRINT_LAYOUT(SCALAR);
        PRINT_LAYOUT(C);
        PRINT_LAYOUT(CHW);
        PRINT_LAYOUT(HWC);
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
enum INFERENCE_ENGINE_1_0_DEPRECATED ColorFormat : uint32_t {
    RAW = 0u,  ///< Plain blob (default), no extra color processing required
    RGB,       ///< RGB color format
    BGR,       ///< BGR color format, default in OpenVINO
    RGBX,      ///< RGBX color format with X ignored during inference
    BGRX,      ///< BGRX color format with X ignored during inference
};

/**
 * @brief Prints a string representation of InferenceEngine::ColorFormat to a stream
 * @param out An output stream to send to
 * @param fmt A color format value to print to a stream
 * @return A reference to the `out` stream
 */
INFERENCE_ENGINE_1_0_DEPRECATED inline std::ostream& operator<<(std::ostream& out, const ColorFormat& fmt) {
    switch (fmt) {
#define PRINT_COLOR_FORMAT(name) \
    case name:                   \
        out << #name;            \
        break;

        PRINT_COLOR_FORMAT(RAW);
        PRINT_COLOR_FORMAT(RGB);
        PRINT_COLOR_FORMAT(BGR);
        PRINT_COLOR_FORMAT(RGBX);
        PRINT_COLOR_FORMAT(BGRX);
#undef PRINT_COLOR_FORMAT

    default:
        out << static_cast<uint32_t>(fmt);
        break;
    }
    return out;
}

/**
 * @struct InferenceEngineProfileInfo
 * @brief Represents basic inference profiling information per layer.
 *
 * If the layer is executed using tiling, the sum time per each tile is indicated as the total execution time.
 * Due to parallel execution, the total execution time for all layers might be greater than the total inference time.
 */
struct INFERENCE_ENGINE_1_0_DEPRECATED InferenceEngineProfileInfo {
    /**
     * @brief Defines the general status of the layer
     */
    enum INFERENCE_ENGINE_1_0_DEPRECATED LayerStatus {
        NOT_RUN,        //!< A layer is not executed
        OPTIMIZED_OUT,  //!< A layer is optimized out during graph optimization phase
        EXECUTED        //!< A layer is executed
    };

    /**
     * @brief Defines a layer status
     */
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
enum INFERENCE_ENGINE_1_0_DEPRECATED StatusCode : int {
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
    NETWORK_NOT_READ = -12,
    INFER_CANCELLED = -13
};

/**
 * @struct ResponseDesc
 * @brief  Represents detailed information for an error
 */
struct INFERENCE_ENGINE_1_0_DEPRECATED ResponseDesc {
    /**
     * @brief A character buffer that holds the detailed information for an error.
     */
    char msg[4096] = {};
};

/**
 * @brief Response structure encapsulating information about supported layer
 */
struct INFERENCE_ENGINE_1_0_DEPRECATED QueryNetworkResult {
    /**
     * @brief A map of supported layers:
     * - key - a layer name
     * - value - a device name on which layer is assigned
     */
    std::map<std::string, std::string> supportedLayersMap;

    /**
     * @brief A status code
     */
    StatusCode rc = OK;

    /**
     * @brief Response message
     */
    ResponseDesc resp;
};

/**
 * @brief A collection that contains string as key, and const Data smart pointer as value
 */
using ConstOutputsDataMap = std::map<std::string, CDataPtr>;

/**
 * @brief A collection that contains string as key, and Data smart pointer as value
 */
using OutputsDataMap = std::map<std::string, DataPtr>;

namespace details {
struct INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(InferenceEngineException)
    : public std::runtime_error {
    using std::runtime_error::runtime_error;
    bool hasStatus() const {
        return true;
    }
    StatusCode getStatus() const;
};
}  // namespace details

/**
 * @brief Base Inference Engine exception class
 */
IE_SUPPRESS_DEPRECATED_START
struct INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(Exception)
    : public details::InferenceEngineException {
    using InferenceEngineException::InferenceEngineException;
};
IE_SUPPRESS_DEPRECATED_END

/// @cond
namespace details {
template <typename ExceptionType>
struct ExceptionTraits;

template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED ExceptionTraits<InferenceEngineException> {
    static const char* string() {
        return "";
    }
};
}  // namespace details

#define INFERENCE_ENGINE_DECLARE_EXCEPTION(ExceptionType, statusCode)                      \
    struct INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(ExceptionType) final \
        : public InferenceEngine::Exception {                                              \
        using Exception::Exception;                                                        \
    };                                                                                     \
    namespace details {                                                                    \
    template <>                                                                            \
    struct INFERENCE_ENGINE_1_0_DEPRECATED ExceptionTraits<ExceptionType> {                \
        static const char* string() {                                                      \
            return "[ " #statusCode " ]";                                                  \
        }                                                                                  \
    };                                                                                     \
    }
/// @endcond

/** @brief This class represents StatusCode::GENERAL_ERROR exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(GeneralError, GENERAL_ERROR)

/** @brief This class represents StatusCode::NOT_IMPLEMENTED exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(NotImplemented, NOT_IMPLEMENTED)

/** @brief This class represents StatusCode::NETWORK_NOT_LOADED exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(NetworkNotLoaded, NETWORK_NOT_LOADED)

/** @brief This class represents StatusCode::PARAMETER_MISMATCH exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(ParameterMismatch, PARAMETER_MISMATCH)

/** @brief This class represents StatusCode::NOT_FOUND exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(NotFound, NOT_FOUND)

/** @brief This class represents StatusCode::OUT_OF_BOUNDS exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(OutOfBounds, OUT_OF_BOUNDS)

/** @brief This class represents StatusCode::UNEXPECTED exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(Unexpected, UNEXPECTED)

/** @brief This class represents StatusCode::REQUEST_BUSY exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(RequestBusy, REQUEST_BUSY)

/** @brief This class represents StatusCode::RESULT_NOT_READY exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(ResultNotReady, RESULT_NOT_READY)

/** @brief This class represents StatusCode::NOT_ALLOCATED exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(NotAllocated, NOT_ALLOCATED)

/** @brief This class represents StatusCode::INFER_NOT_STARTED exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(InferNotStarted, INFER_NOT_STARTED)

/** @brief This class represents StatusCode::NETWORK_NOT_READ exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(NetworkNotRead, NETWORK_NOT_READ)

/** @brief This class represents StatusCode::INFER_CANCELLED exception */
INFERENCE_ENGINE_DECLARE_EXCEPTION(InferCancelled, INFER_CANCELLED)

/**
 * @private
 */
#undef INFERENCE_ENGINE_DECLARE_EXCEPTION

// TODO: Move this section out of public API
namespace details {

/**
 * @brief Rethrow a copy of exception. UShould be used in catch blocks
 */
[[noreturn]] INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CPP(void) Rethrow();

/**
 * @brief Tag struct used to throw exception
 */
#ifndef NDEBUG
template <typename ExceptionType>
struct INFERENCE_ENGINE_1_0_DEPRECATED ThrowNow final {
    const char* const file;
    const int line;

    [[noreturn]] static void create(const std::ostream& ostream, const char* file, int line) {
        std::stringstream stream;
        stream << '\n' << file << ':' << line << ' ';
        stream << ExceptionTraits<ExceptionType>::string() << ' ' << ostream.rdbuf();
        throw ExceptionType{stream.str()};
    }

    [[noreturn]] void operator<<=(const std::ostream& ostream) {
        create(ostream, file, line);
    }
};
#else
template <typename ExceptionType>
struct INFERENCE_ENGINE_1_0_DEPRECATED ThrowNow final {
    [[noreturn]] static void create(const std::ostream& ostream) {
        std::stringstream stream;
        stream << ExceptionTraits<ExceptionType>::string() << ' ' << ostream.rdbuf();
        throw ExceptionType{stream.str()};
    }

    [[noreturn]] void operator<<=(const std::ostream& ostream) {
        create(ostream);
    }
};
#endif

/// @cond
#ifndef NDEBUG
#    define IE_LOCATION       '\n' << __FILE__ << ':' << __LINE__ << ' '
#    define IE_LOCATION_PARAM __FILE__, __LINE__
#else
#    define IE_LOCATION ""
#    define IE_LOCATION_PARAM
#endif  // NDEBUG

// WARNING: DO NOT USE THIS MACRO! Use openvino/util/pp.hpp macro library
#define IE_PP_EXPAND(X)             X
#define IE_PP_NARG(...)             IE_PP_EXPAND(IE_PP_NARG_(__VA_ARGS__, IE_PP_RSEQ_N()))
#define IE_PP_NARG_(...)            IE_PP_EXPAND(IE_PP_ARG_N(__VA_ARGS__))
#define IE_PP_ARG_N(_0, _1, N, ...) N
#define IE_PP_RSEQ_N()              0, 1, 0
#define IE_PP_NO_ARGS(NAME)         ,
#define IE_PP_CAT3_(x, y, z)        x##y##z
#define IE_PP_CAT3(x, y, z)         IE_PP_CAT3_(x, y, z)
#define IE_PP_OVERLOAD(NAME, ...) \
    IE_PP_EXPAND(IE_PP_CAT3(NAME, _, IE_PP_EXPAND(IE_PP_NARG(IE_PP_NO_ARGS __VA_ARGS__(NAME))))(__VA_ARGS__))
// ENDWARNING

#define IE_THROW_0() \
    (InferenceEngine::details::ThrowNow<InferenceEngine::GeneralError>{IE_LOCATION_PARAM}) <<= std::stringstream {}

#define IE_THROW_1(ExceptionType) \
    (InferenceEngine::details::ThrowNow<InferenceEngine::ExceptionType>{IE_LOCATION_PARAM}) <<= std::stringstream {}
/// @endcond

/**
 * @def IE_THROW
 * @brief A macro used to throw specified exception with a description
 */
#define IE_THROW(...) IE_PP_OVERLOAD(IE_THROW, __VA_ARGS__)

/**
 * @def IE_ASSERT
 * @brief Uses assert() function if NDEBUG is not defined, InferenceEngine exception otherwise
 */
#ifdef NDEBUG
#    define IE_ASSERT(EXPRESSION) \
        if (!(EXPRESSION))        \
        IE_THROW(GeneralError) << " AssertionError " #EXPRESSION
#else
/**
 * @private
 */
struct NullStream {
    template <typename T>
    NullStream& operator<<(const T&) noexcept {
        return *this;
    }
};

#    define IE_ASSERT(EXPRESSION) \
        assert((EXPRESSION));     \
        InferenceEngine::details::NullStream()
#endif  // NDEBUG

/// @cond
#define THROW_IE_EXCEPTION                                                                                          \
    (InferenceEngine::details::ThrowNow<InferenceEngine::details::InferenceEngineException>{IE_LOCATION_PARAM}) <<= \
        std::stringstream {}

#define IE_EXCEPTION_CASE(TYPE_ALIAS, STATUS_CODE, EXCEPTION_TYPE, ...) \
    case InferenceEngine::STATUS_CODE: {                                \
        using InferenceEngine::EXCEPTION_TYPE;                          \
        using TYPE_ALIAS = EXCEPTION_TYPE;                              \
        __VA_ARGS__;                                                    \
    } break;
/// @endcond

/**
 * @def IE_EXCEPTION_SWITCH
 * @brief Generate Switch statement over error codes adn maps them to coresponding exceptions type
 */
#define IE_EXCEPTION_SWITCH(STATUS, TYPE_ALIAS, ...)                                      \
    switch (STATUS) {                                                                     \
        IE_EXCEPTION_CASE(TYPE_ALIAS, GENERAL_ERROR, GeneralError, __VA_ARGS__)           \
        IE_EXCEPTION_CASE(TYPE_ALIAS, NOT_IMPLEMENTED, NotImplemented, __VA_ARGS__)       \
        IE_EXCEPTION_CASE(TYPE_ALIAS, NETWORK_NOT_LOADED, NetworkNotLoaded, __VA_ARGS__)  \
        IE_EXCEPTION_CASE(TYPE_ALIAS, PARAMETER_MISMATCH, ParameterMismatch, __VA_ARGS__) \
        IE_EXCEPTION_CASE(TYPE_ALIAS, NOT_FOUND, NotFound, __VA_ARGS__)                   \
        IE_EXCEPTION_CASE(TYPE_ALIAS, OUT_OF_BOUNDS, OutOfBounds, __VA_ARGS__)            \
        IE_EXCEPTION_CASE(TYPE_ALIAS, UNEXPECTED, Unexpected, __VA_ARGS__)                \
        IE_EXCEPTION_CASE(TYPE_ALIAS, REQUEST_BUSY, RequestBusy, __VA_ARGS__)             \
        IE_EXCEPTION_CASE(TYPE_ALIAS, RESULT_NOT_READY, ResultNotReady, __VA_ARGS__)      \
        IE_EXCEPTION_CASE(TYPE_ALIAS, NOT_ALLOCATED, NotAllocated, __VA_ARGS__)           \
        IE_EXCEPTION_CASE(TYPE_ALIAS, INFER_NOT_STARTED, InferNotStarted, __VA_ARGS__)    \
        IE_EXCEPTION_CASE(TYPE_ALIAS, NETWORK_NOT_READ, NetworkNotRead, __VA_ARGS__)      \
        IE_EXCEPTION_CASE(TYPE_ALIAS, INFER_CANCELLED, InferCancelled, __VA_ARGS__)       \
    default:                                                                              \
        IE_ASSERT(!"Unreachable");                                                        \
    }

}  // namespace details
}  // namespace InferenceEngine

#if defined(_WIN32) && !defined(__GNUC__)
#    define __PRETTY_FUNCTION__ __FUNCSIG__
#else
#    define __PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif
IE_SUPPRESS_DEPRECATED_END
