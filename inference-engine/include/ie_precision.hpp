// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides class for describing precision of data
 * @file ie_precision.hpp
 */
#pragma once
#include <unordered_map>
#include <string>
#include "details/ie_exception.hpp"

namespace InferenceEngine {

/**
 * @brief This class holds precision value and provides precision related operations
 */
class Precision {
public:
    /** Enum to specify of different  */
    enum ePrecision : uint8_t {
        UNSPECIFIED = 255, /**< Unspecified value. Used by default */
        MIXED = 0,  /**< Mixed value. Can be received from network. No applicable for tensors */
        FP32 = 10,  /**< 32bit floating point value */
        FP16 = 11,  /**< 16bit floating point value */
        Q78 = 20,   /**< 16bit specific signed fixed point precision */
        I16 = 30,   /**< 16bit signed integer value */
        U8 = 40,    /**< 8bit unsigned integer value */
        I8 = 50,    /**< 8bit signed integer value */
        U16 = 60,   /**< 16bit unsigned integer value */
        I32 = 70,   /**< 32bit signed integer value */
        I64 = 72,   /**< 64bit signed integer value */
        BIN = 71,   /**< 1bit integer value */
        CUSTOM = 80 /**< custom precision has it's own name and size of elements */
    };

private:
    struct PrecisionInfo {
        /** @brief Size of underlined element */
        size_t bitsSize = 0;

        /** @brief Null terminated string with precision name */
        const char *name = "UNSPECIFIED";

        bool isFloat     = false;
        ePrecision value = Precision::UNSPECIFIED;
    };
    PrecisionInfo precisionInfo;

public:
    /** @brief Default constructor */
    Precision()  = default;

    /** @brief Constructor with specified precision */
    Precision(const Precision::ePrecision  value) {  // NOLINT
        precisionInfo = getPrecisionInfo(value);
    }

    /**
     * @brief Custom precision constructor
     *
     * @param bitsSize size of elements
     * @param name optional name string, used in serialisation
     */
    explicit Precision(size_t bitsSize, const char * name = nullptr) {
        if (bitsSize == 0) {
            THROW_IE_EXCEPTION << "Precision with 0 elements size not supported";
        }
        precisionInfo.bitsSize = bitsSize;
        if (name == nullptr) {
            precisionInfo.name = "CUSTOM";
        } else {
            precisionInfo.name = name;
        }
        precisionInfo.value = CUSTOM;
    }

    /** @brief Creates custom precision with specific underlined type */
    template <class T>
    static Precision fromType(const char * typeName = nullptr) {
        return Precision(8 * sizeof(T), typeName == nullptr ? typeid(T).name() : typeName);
    }

    /** @brief checks whether given storage class T can be used to store objects of current precision */
    template <class T>
    bool hasStorageType(const char * typeName = nullptr) const noexcept {
        try {
            if (precisionInfo.value != BIN) {
                if (sizeof(T) != size()) {
                    return false;
                }
            }
#define CASE(x, y) case x: return std::is_same<T, y>()
#define CASE2(x, y1, y2) case x: return std::is_same<T, y1>() || std::is_same<T, y2>()

            switch (precisionInfo.value) {
                CASE(FP32, float);
                CASE2(FP16, int16_t, uint16_t);
                CASE(I16, int16_t);
                CASE(I32, int32_t);
                CASE(I64, int64_t);
                CASE(U16, uint16_t);
                CASE(U8, uint8_t);
                CASE(I8, int8_t);
                CASE2(Q78, int16_t, uint16_t);
                CASE2(BIN, int8_t, uint8_t);
                default :
                    return areSameStrings(name(), typeName == nullptr ? typeid(T).name() : typeName);
#undef CASE
#undef CASE2
            }
        } catch (...) {
            return false;
        }
    }

    /** @brief Equality operator with Precision object */
    bool operator == (const Precision  & p) const noexcept {
        return precisionInfo.value == p &&
            precisionInfo.bitsSize == p.precisionInfo.bitsSize &&
            areSameStrings(precisionInfo.name, p.precisionInfo.name);
    }

    /** @brief Equality operator with ePrecision enum value */
    bool operator == (const ePrecision  p) const noexcept {
        return precisionInfo.value == p;
    }

    /** @brief Inequality operator with ePrecision enum value */
    bool operator != (const ePrecision   p) const noexcept {
        return precisionInfo.value != p;
    }

    /** @brief Assignment operator with ePrecision enum value */
    Precision & operator = (const ePrecision p) noexcept {
        precisionInfo = getPrecisionInfo(p);
        return *this;
    }

    /** @brief Cast operator to a bool */
    explicit operator bool() const noexcept {
        return precisionInfo.value != UNSPECIFIED;
    }

    /** @brief Logical negation operator */
    bool operator !() const noexcept {
        return precisionInfo.value == UNSPECIFIED;
    }

    /** @brief Cast operator to a ePrecision */
    operator Precision::ePrecision  () const noexcept {
        return precisionInfo.value;
    }

    /** @brief Getter of precision name */
    const char *name() const noexcept {
        return precisionInfo.name;
    }

    /** @brief Creates from string with precision name */
    static Precision FromStr(const std::string &str) {
        static std::unordered_map<std::string, ePrecision > names = {
#define     PRECISION_NAME(s) {#s, s}
            PRECISION_NAME(Q78),
            PRECISION_NAME(U8),
            PRECISION_NAME(I8),
            PRECISION_NAME(I16),
            PRECISION_NAME(I32),
            PRECISION_NAME(I64),
            PRECISION_NAME(U16),
            PRECISION_NAME(FP32),
            PRECISION_NAME(FP16),
            PRECISION_NAME(MIXED),
            PRECISION_NAME(BIN),
#undef      PRECISION_NAME
        };
        auto i = names.find(str);
        return i == names.end() ? Precision() : Precision(i->second);
    }

    /**
     * @brief Returns size of single element of that precision in bits
     *
     * @returns Number of bits per element
     */
    size_t size() const {
        if (precisionInfo.bitsSize == 0) {
            THROW_IE_EXCEPTION << " cannot estimate element if precision is " << precisionInfo.name;
        }
        return precisionInfo.bitsSize >> 3;
    }

    /** @brief Checks if it is a floating point */
    bool is_float() const noexcept {
        return precisionInfo.isFloat;
    }

 protected:
    /**
     * @brief Returns PrecisionInfo by its name
     *
     * @param name Name of precision
     */
    template<Precision::ePrecision precision>
    static PrecisionInfo makePrecisionInfo(const char * name);

    /**
     * @brief Compare two c-strings
     *
     * @param l Const pointer to first string
     * @param r Const pointer to another string
     * @returns True if strings are the same
     */
    static bool areSameStrings(const char *l, const char *r) noexcept {
        if (l == r)
            return true;

        if (l == nullptr || r == nullptr)
            return false;

        for (; *l && *r; l++, r++) {
            if (*l != *r) return false;
        }
        return *l == *r;
    }

    /**
     * @brief Return PrecisionInfo
     */
    static PrecisionInfo getPrecisionInfo(ePrecision v) {
#define CASE(x) case x: return makePrecisionInfo<x>(#x);
        switch (v) {
            CASE(FP32);
            CASE(FP16);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            CASE(U16);
            CASE(U8);
            CASE(I8);
            CASE(Q78);
            CASE(MIXED);
            CASE(BIN);
            default : return makePrecisionInfo<UNSPECIFIED>("UNSPECIFIED");
#undef CASE
        }
    }
};

/**
 * @brief Particular precision traits
 */
template<Precision::ePrecision p>
struct PrecisionTrait {
};

/** @cond INTERNAL */
template<>
struct PrecisionTrait<Precision::FP32> {
    using value_type = float;
};

template<>
struct PrecisionTrait<Precision::FP16> {
    using value_type = int16_t;
};
template<>
struct PrecisionTrait<Precision::Q78> {
    using value_type = uint16_t;
};
template<>
struct PrecisionTrait<Precision::I16> {
    using value_type = int16_t;
};
template<>
struct PrecisionTrait<Precision::U16> {
    using value_type = uint16_t;
};
template<>
struct PrecisionTrait<Precision::U8> {
    using value_type = uint8_t;
};
template<>
struct PrecisionTrait<Precision::I8> {
    using value_type = int8_t;
};
template<>
struct PrecisionTrait<Precision::I32> {
    using value_type = int32_t;
};
template<>
struct PrecisionTrait<Precision::I64> {
    using value_type = int64_t;
};
template<>
struct PrecisionTrait<Precision::BIN> {
    using value_type = int8_t;
};

template<class T>
inline uint8_t type_size_or_zero() {
    return sizeof(T);
}

template<>
struct PrecisionTrait<Precision::UNSPECIFIED> {
    using value_type = void;
};

template<>
struct PrecisionTrait<Precision::MIXED> : PrecisionTrait<Precision::UNSPECIFIED>{
};

template<>
inline uint8_t type_size_or_zero<void>() {
    return 0;
}

template<Precision::ePrecision T>
inline typename std::enable_if<std::is_same<
    std::integral_constant<Precision::ePrecision, Precision::FP16>,
    std::integral_constant<Precision::ePrecision, T>>::value, bool>::type is_floating() {
    return true;
}

template<Precision::ePrecision T>
inline typename std::enable_if<!std::is_same<
    std::integral_constant<Precision::ePrecision, Precision::FP16>,
    std::integral_constant<Precision::ePrecision, T>>::value, bool>::type is_floating() {
    return std::is_floating_point<typename PrecisionTrait<T>::value_type>::value;
}

template<Precision::ePrecision precision>
inline Precision::PrecisionInfo Precision::makePrecisionInfo(const char *name) {
    Precision::PrecisionInfo info;
    info.name = name;

    size_t nBits = precision == BIN ? 1 : 8;
    info.bitsSize = nBits * type_size_or_zero<typename PrecisionTrait<precision>::value_type>();
    info.isFloat = is_floating<precision>();
    info.value = precision;
    return info;
}

inline std::ostream & operator << (std::ostream &out, const InferenceEngine::Precision & p) {
    return out << p.name();
}

inline std::ostream & operator << (std::ostream &out, const InferenceEngine::Precision::ePrecision & p) {
    return out << Precision(p).name();
}

inline constexpr uint32_t getPrecisionMask(InferenceEngine::Precision::ePrecision precision1,
                                           InferenceEngine::Precision::ePrecision precision2,
                                           InferenceEngine::Precision::ePrecision precision3 = InferenceEngine::Precision::MIXED,
                                           InferenceEngine::Precision::ePrecision precision4 = InferenceEngine::Precision::MIXED) {
    return (precision1) | (precision2 << 8) | (precision3 << 16) | (precision4 << 24);
}

/** @endcond */

}  // namespace InferenceEngine
