// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides class for describing precision of data
 *
 * @file ie_precision.hpp
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

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "ie_common.h"

namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START

/**
 * @brief This class holds precision value and provides precision related operations
 */
class INFERENCE_ENGINE_1_0_DEPRECATED Precision {
public:
    /** Enum to specify of different  */
    enum ePrecision : uint8_t {
        UNSPECIFIED = 255, /**< Unspecified value. Used by default */
        MIXED = 0,         /**< Mixed value. Can be received from network. No applicable for tensors */
        FP32 = 10,         /**< 32bit floating point value */
        FP16 = 11,         /**< 16bit floating point value, 5 bit for exponent, 10 bit for mantisa */
        BF16 = 12,         /**< 16bit floating point value, 8 bit for exponent, 7 bit for mantisa*/
        FP64 = 13,         /**< 64bit floating point value */
        NF4 = 14,          /**< 4bit normalized float value */
        Q78 = 20,          /**< 16bit specific signed fixed point precision */
        I16 = 30,          /**< 16bit signed integer value */
        U4 = 39,           /**< 4bit unsigned integer value */
        U8 = 40,           /**< 8bit unsigned integer value */
        I4 = 49,           /**< 4bit signed integer value */
        I8 = 50,           /**< 8bit signed integer value */
        U16 = 60,          /**< 16bit unsigned integer value */
        I32 = 70,          /**< 32bit signed integer value */
        U32 = 74,          /**< 32bit unsigned integer value */
        I64 = 72,          /**< 64bit signed integer value */
        U64 = 73,          /**< 64bit unsigned integer value */
        BIN = 71,          /**< 1bit integer value */
        BOOL = 41,         /**< 8bit bool type */
        STRING = 79,       /**< string type, most likely std::string in C++, TODO: choose valid number */
        CUSTOM = 80        /**< custom precision has it's own name and size of elements */
    };

private:
    struct PrecisionInfo {
        /** @brief Size of underlined element */
        size_t bitsSize = 0;

        /** @brief Null terminated string with precision name */
        const char* name = "UNSPECIFIED";

        bool isFloat = false;
        ePrecision value = Precision::UNSPECIFIED;
    };
    PrecisionInfo precisionInfo;

public:
    /** @brief Default constructor */
    Precision() = default;

    /**
     * @brief Constructor with specified precision
     * @param value A value of ePrecision to create an object from
     */
    Precision(const Precision::ePrecision value) {
        precisionInfo = getPrecisionInfo(value);
    }

    /**
     * @brief Custom precision constructor
     *
     * @param bitsSize size of elements
     * @param name optional: name string, used in serialisation
     */
    explicit Precision(size_t bitsSize, const char* name = nullptr) {
        if (bitsSize == 0) {
            IE_THROW() << "Precision with 0 elements size not supported";
        }
        precisionInfo.bitsSize = bitsSize;
        if (name == nullptr) {
            precisionInfo.name = "CUSTOM";
        } else {
            precisionInfo.name = name;
        }
        precisionInfo.value = CUSTOM;
    }

    /**
     * @brief Creates custom precision with specific underlined type
     * @param typeName A string name of precision
     * @return Precision converted from string name
     */
    template <class T>
    static Precision fromType(const char* typeName = nullptr) {
        return Precision(8 * sizeof(T), typeName == nullptr ? typeid(T).name() : typeName);
    }

    /**
     * @brief checks whether given storage class T can be used to store objects of current precision
     * @param typeName A string name of precision
     * @return `true` if `typeName` has underlaying storage type
     */
    template <class T>
    bool hasStorageType(const char* typeName = nullptr) const noexcept {
        try {
#define CASE(x, y) \
    case x:        \
        return std::is_same<T, y>()
#define CASE2(x, y1, y2) \
    case x:              \
        return std::is_same<T, y1>() || std::is_same<T, y2>()

            switch (precisionInfo.value) {
                CASE(FP32, float);
                CASE(FP64, double);
                CASE2(FP16, int16_t, uint16_t);
                CASE2(BF16, int16_t, uint16_t);
                CASE(NF4, int8_t);
                CASE2(I4, int8_t, uint8_t);
                CASE(I8, int8_t);
                CASE(I16, int16_t);
                CASE(I32, int32_t);
                CASE(I64, int64_t);
                CASE(U4, uint8_t);
                CASE(U8, uint8_t);
                CASE(U16, uint16_t);
                CASE(U32, uint32_t);
                CASE(U64, uint64_t);
                CASE(BOOL, uint8_t);
                CASE2(Q78, int16_t, uint16_t);
                CASE2(BIN, int8_t, uint8_t);
                CASE(STRING, std::string);
            default:
                return areSameStrings(name(), typeName == nullptr ? typeid(T).name() : typeName);
#undef CASE
#undef CASE2
            }
        } catch (...) {
            return false;
        }
    }

    /**
     * @brief Equality operator with Precision object
     * @param p A value of Precision to compare with
     * @return `true` if values represent the same precisions, `false` otherwise
     */
    bool operator==(const Precision& p) const noexcept {
        return precisionInfo.value == p && precisionInfo.bitsSize == p.precisionInfo.bitsSize &&
               areSameStrings(precisionInfo.name, p.precisionInfo.name);
    }

    /**
     * @brief Inequality operator with Precision object
     * @param p A value of Precision to compare with
     * @return `true` if values represent different precisions, `false` otherwise
     */
    bool operator!=(const Precision& p) const noexcept {
        return !(*this == p);
    }

    /**
     * @brief Equality operator with ePrecision enum value
     * @param p A value of ePrecision to compare with
     * @return `true` if values represent the same precisions, `false` otherwise
     */
    bool operator==(const ePrecision p) const noexcept {
        return precisionInfo.value == p;
    }

    /**
     * @brief Inequality operator with ePrecision enum value
     * @param p A value of ePrecision to compare with
     * @return `true` if values represent different precisions, `false` otherwise
     */
    bool operator!=(const ePrecision p) const noexcept {
        return precisionInfo.value != p;
    }

    /**
     * @brief Assignment operator with ePrecision enum value
     * @param p A value of ePrecision enumeration
     * @return A Precision instance
     */
    Precision& operator=(const ePrecision p) noexcept {
        precisionInfo = getPrecisionInfo(p);
        return *this;
    }

    /**
     * @brief Cast operator to a bool
     * @return `true` if precision is specified, `false` otherwise
     */
    explicit operator bool() const noexcept {
        return precisionInfo.value != UNSPECIFIED;
    }

    /**
     * @brief Logical negation operator
     * @return `true` if precision is NOT specified, `false` otherwise
     */
    bool operator!() const noexcept {
        return precisionInfo.value == UNSPECIFIED;
    }

    /**
     * @brief Cast operator to a ePrecision
     * @return A casted value of Precision::ePrecision enumeration
     */
    operator Precision::ePrecision() const noexcept {
        return precisionInfo.value;
    }

    /**
     * @brief Gets the precision value of type ePrecision.
     * @return The preccision value.
     */
    constexpr uint8_t getPrecVal() const noexcept {
        return precisionInfo.value;
    }

    /**
     * @brief Getter of precision name
     * @return A string representing precision name
     */
    const char* name() const noexcept {
        return precisionInfo.name;
    }

    /**
     * @brief Creates Precision from string with precision name
     * @param str A string representing precision
     * @return Precision created from string representation
     */
    static Precision FromStr(const std::string& str) {
        static const std::unordered_map<std::string, ePrecision> names = {
#define PRECISION_NAME(s) {#s, s}
            PRECISION_NAME(Q78),   PRECISION_NAME(BOOL), PRECISION_NAME(BF16), PRECISION_NAME(I4),
            PRECISION_NAME(I8),    PRECISION_NAME(I16),  PRECISION_NAME(I32),  PRECISION_NAME(I64),
            PRECISION_NAME(U4),    PRECISION_NAME(U8),   PRECISION_NAME(U16),  PRECISION_NAME(U32),
            PRECISION_NAME(U64),   PRECISION_NAME(FP32), PRECISION_NAME(FP64), PRECISION_NAME(FP16),
            PRECISION_NAME(MIXED), PRECISION_NAME(NF4),  PRECISION_NAME(STRING), PRECISION_NAME(BIN),
#undef PRECISION_NAME
        };
        auto i = names.find(str);
        return i == names.end() ? Precision() : Precision(i->second);
    }

    /**
     * @brief Returns size of single element of that precision in bytes
     * @returns Number of bytes per element
     */
    size_t size() const {
        return (bitsSize() + 7) >> 3;
    }

    /**
     * @brief Returns size of single element of that precision in bits
     * @returns Number of bits per element
     */
    size_t bitsSize() const {
        if (precisionInfo.bitsSize == 0) {
            IE_THROW() << " cannot estimate element if precision is " << precisionInfo.name;
        }
        return precisionInfo.bitsSize;
    }

    /**
     * @brief Checks if it is a floating point value
     * @return True if precision is float point, `false` otherwise
     */
    bool is_float() const noexcept {
        return precisionInfo.isFloat;
    }

    /**
     * @brief Checks if it is a signed value
     * @return True if precision is signed, `false` otherwise
     */
    bool isSigned() const noexcept {
        return (precisionInfo.value == Precision::UNSPECIFIED) || (precisionInfo.value == Precision::MIXED) ||
               (precisionInfo.value == Precision::FP32) || (precisionInfo.value == Precision::FP64) ||
               (precisionInfo.value == Precision::FP16) || (precisionInfo.value == Precision::Q78) ||
               (precisionInfo.value == Precision::I16) || (precisionInfo.value == Precision::I8) ||
               (precisionInfo.value == Precision::I32) || (precisionInfo.value == Precision::I64) ||
               (precisionInfo.value == Precision::BIN) || (precisionInfo.value == Precision::BF16) ||
               (precisionInfo.value == Precision::CUSTOM) || (precisionInfo.value == Precision::I4) ||
               (precisionInfo.value == Precision::NF4);
    }

protected:
    /**
     * @brief Creates PrecisionInfo by @p precision with a specified name
     * @tparam precision A precision to create PrecisionInfo for
     * @param name Name of precision
     * @return A PrecisionInfo object
     */
    template <Precision::ePrecision precision>
    static PrecisionInfo makePrecisionInfo(const char* name);

    /**
     * @brief Compare two c-strings
     *
     * @param l Const pointer to first string
     * @param r Const pointer to another string
     * @returns True if strings are the same
     */
    static bool areSameStrings(const char* l, const char* r) noexcept {
        if (l == r)
            return true;

        if (l == nullptr || r == nullptr)
            return false;

        for (; *l && *r; l++, r++) {
            if (*l != *r)
                return false;
        }
        return *l == *r;
    }

    /**
     * @brief Creates PrecisionInfo based on ePrecision
     * @param v A value of ePrecision emuneration
     * @return Precision info object
     */
    static PrecisionInfo getPrecisionInfo(ePrecision v) {
#define CASE(x) \
    case x:     \
        return makePrecisionInfo<x>(#x);
        switch (v) {
            CASE(FP32);
            CASE(FP64);
            CASE(FP16);
            CASE(BF16);
            CASE(NF4);
            CASE(I4);
            CASE(I8);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            CASE(U4);
            CASE(U8);
            CASE(U16);
            CASE(U32);
            CASE(U64);
            CASE(Q78);
            CASE(MIXED);
            CASE(BIN);
            CASE(BOOL);
            CASE(STRING);
        default:
            return makePrecisionInfo<UNSPECIFIED>("UNSPECIFIED");
#undef CASE
        }
    }
};

/**
 * @brief Particular precision traits
 */
template <Precision::ePrecision p>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait {};

/** @cond INTERNAL */
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::FP32> {
    using value_type = float;
    enum { is_float = true };
};

template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::FP64> {
    using value_type = double;
    enum { is_float = true };
};

template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::FP16> {
    using value_type = int16_t;
    enum { is_float = true };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::BF16> {
    using value_type = int16_t;
    enum { is_float = true };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::Q78> {
    using value_type = uint16_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::I16> {
    using value_type = int16_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::U16> {
    using value_type = uint16_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::U4> {
    using value_type = uint8_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::U8> {
    using value_type = uint8_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::I4> {
    using value_type = int8_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::I8> {
    using value_type = int8_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::BOOL> {
    using value_type = uint8_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::I32> {
    using value_type = int32_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::U32> {
    using value_type = uint32_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::I64> {
    using value_type = int64_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::U64> {
    using value_type = uint64_t;
    enum { is_float = false };
};
template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::BIN> {
    using value_type = int8_t;
    enum { is_float = false };
};

template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::NF4> {
    using value_type = int8_t;
    enum { is_float = false };
};

template <class T>
INFERENCE_ENGINE_1_0_DEPRECATED inline uint8_t type_size_or_zero() {
    return sizeof(T);
}

template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::UNSPECIFIED> {
    using value_type = void;
    enum { is_float = false };
};

template <>
struct PrecisionTrait<Precision::STRING> {
    using value_type = std::string;
    enum { is_float = false };
};

template <>
struct INFERENCE_ENGINE_1_0_DEPRECATED PrecisionTrait<Precision::MIXED> : PrecisionTrait<Precision::UNSPECIFIED> {};

template <>
INFERENCE_ENGINE_1_0_DEPRECATED inline uint8_t type_size_or_zero<void>() {
    return 0;
}

template <Precision::ePrecision precision>
INFERENCE_ENGINE_1_0_DEPRECATED inline Precision::PrecisionInfo Precision::makePrecisionInfo(const char* name) {
    Precision::PrecisionInfo info;
    info.name = name;

    size_t nBits = precision == BIN ? 1 : (precision == U4 || precision == I4 || precision == NF4) ? 4 : 8;
    info.bitsSize = nBits * type_size_or_zero<typename PrecisionTrait<precision>::value_type>();
    //std::cerr << "makePrecisionInfo(" << name << ").info.bitsSize = " << info.bitsSize << "\n";
    if(name == std::string("UNSPECIFIED")) {
        //std::cerr << "UNSPECIFIED\n";
    }
    info.isFloat = PrecisionTrait<precision>::is_float;
    info.value = precision;
    return info;
}

inline std::ostream& operator<<(std::ostream& out, const InferenceEngine::Precision& p) {
    return out << p.name();
}

inline std::ostream& operator<<(std::ostream& out, const InferenceEngine::Precision::ePrecision& p) {
    return out << Precision(p).name();
}

inline std::ostream& operator<<(std::ostream& os, const std::vector<Precision>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

INFERENCE_ENGINE_1_0_DEPRECATED inline constexpr uint32_t getPrecisionMask(
    InferenceEngine::Precision::ePrecision precision1,
    InferenceEngine::Precision::ePrecision precision2,
    InferenceEngine::Precision::ePrecision precision3 = InferenceEngine::Precision::MIXED,
    InferenceEngine::Precision::ePrecision precision4 = InferenceEngine::Precision::MIXED) {
    return (precision1) | (precision2 << 8) | (precision3 << 16) | (precision4 << 24);
}

/** @endcond */

IE_SUPPRESS_DEPRECATED_END
}  // namespace InferenceEngine
