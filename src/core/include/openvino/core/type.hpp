// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/compatibility.hpp"
#include "openvino/core/core_visibility.hpp"

namespace ov {

// constexpr CRC64 algorithm
namespace util {

constexpr size_t strlen(const char* str) {
    return (*str) == '\0' ? 0 : strlen(str + 1) + 1;
}

constexpr uint64_t char_to_uint64(const char c, size_t to_bit, uint64_t data) {
    return data | (((uint64_t(0) | c) << 8 * to_bit));
}
constexpr uint64_t char_arr_to_uint64(const char* str, size_t len, uint64_t data) {
    return len > 0 ? char_to_uint64(str[0], len - 1, char_arr_to_uint64(str + 1, len - 1, data)) : data;
}

constexpr uint64_t hash_xor_and_shift(uint64_t v, uint64_t shift) {
    return v ^ (v >> shift);
}

constexpr uint64_t _murmur_mul(uint64_t v) {
    return v * uint64_t(0xc6a4a7935bd1e995);
}

constexpr uint64_t _murmur_mul_shift_mul(uint64_t v) {
    // k *= m;
    // k ^= k >> 47;
    // k *= m;
    return _murmur_mul(hash_xor_and_shift(_murmur_mul(v), 47));
}

constexpr uint64_t _murmur_hash_all(uint64_t v, uint64_t seed) {
    // k *= m;
    // k ^= k >> 47;
    // k *= m;
    //
    // h ^= k;
    // h *= m;
    return _murmur_mul(seed ^ _murmur_mul_shift_mul(v));
}

constexpr uint64_t _murmur_last_val(const char* str, size_t len, uint64_t seed) {
    // case 7: h ^= ((uint64_t) data2[6]) << 48;
    // case 6: h ^= ((uint64_t) data2[5]) << 40;
    // case 5: h ^= ((uint64_t) data2[4]) << 32;
    // case 4: h ^= ((uint64_t) data2[3]) << 24;
    // case 3: h ^= ((uint64_t) data2[2]) << 16;
    // case 2: h ^= ((uint64_t) data2[1]) << 8;
    // case 1: h ^= ((uint64_t) data2[0]);
    //         h *= m;
    return len == 0 ? seed : _murmur_last_val(str, len - 1, seed ^ char_to_uint64(str[len - 1], len - 1, 0));
}

constexpr uint64_t _murmur_hash_last(const char* str, size_t len, uint64_t seed) {
    // switch(len & 7)
    // {
    // case 7: h ^= ((uint64_t) data2[6]) << 48;
    // case 6: h ^= ((uint64_t) data2[5]) << 40;
    // case 5: h ^= ((uint64_t) data2[4]) << 32;
    // case 4: h ^= ((uint64_t) data2[3]) << 24;
    // case 3: h ^= ((uint64_t) data2[2]) << 16;
    // case 2: h ^= ((uint64_t) data2[1]) << 8;
    // case 1: h ^= ((uint64_t) data2[0]);
    //         h *= m;
    // };
    //
    // h ^= h >> r;
    // h *= m;
    // h ^= h >> r;
    return hash_xor_and_shift(_murmur_mul_shift_mul(_murmur_last_val(str, len, seed)), 47);
}

constexpr uint64_t _murmur_hash(const char* str, size_t len, uint64_t seed) {
    return len >= 8 ? _murmur_hash(str + 8, len - 8, _murmur_hash_all(char_arr_to_uint64(str, 8, 0), seed))
                    : _murmur_hash_last(str, len, seed);
}

constexpr uint64_t murmur_hash(const char* str, size_t len, uint64_t seed) {
    return _murmur_hash(str, len, seed ^ _murmur_mul(len));
}

}  // namespace util

/// Supports three functions, ov::is_type<Type>, ov::as_type<Type>, and ov::as_type_ptr<Type> for type-safe
/// dynamic conversions via static_cast/static_ptr_cast without using C++ RTTI.
/// Type must have a static type_info member and a virtual get_type_info() member that
/// returns a reference to its type_info member.

/// Type information for a type system without inheritance; instances have exactly one type not
/// related to any other type.
struct OPENVINO_API DiscreteTypeInfo {
    const char* name;
    uint64_t version;
    const char* version_id;
    // A pointer to a parent type info; used for casting and inheritance traversal, not for
    // exact type identification
    const DiscreteTypeInfo* parent;

    DiscreteTypeInfo() = default;

    constexpr DiscreteTypeInfo(const char* _name, uint64_t _version, const DiscreteTypeInfo* _parent = nullptr)
        : name(_name),
          version(_version),
          version_id(nullptr),
          parent(_parent),
          hash_value(compute_hash()) {}

    constexpr DiscreteTypeInfo(const char* _name,
                               uint64_t _version,
                               const char* _version_id,
                               const DiscreteTypeInfo* _parent = nullptr)
        : name(_name),
          version(_version),
          version_id(_version_id),
          parent(_parent),
          hash_value(compute_hash()) {}

    bool is_castable(const DiscreteTypeInfo& target_type) const;

    std::string get_version() const;

    // For use as a key
    bool operator<(const DiscreteTypeInfo& b) const;
    bool operator<=(const DiscreteTypeInfo& b) const;
    bool operator>(const DiscreteTypeInfo& b) const;
    bool operator>=(const DiscreteTypeInfo& b) const;
    bool operator==(const DiscreteTypeInfo& b) const;
    bool operator!=(const DiscreteTypeInfo& b) const;

    operator std::string() const;

    size_t hash() const;

private:
    uint64_t hash_value;

    constexpr uint64_t compute_hash() const {
        return util::murmur_hash(
            name,
            util::strlen(name),
            version_id == nullptr ? version : util::murmur_hash(version_id, util::strlen(version_id), version));
    }
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const DiscreteTypeInfo& info);

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to a
/// Type*/shared_ptr<Type>
OPENVINO_SUPPRESS_DEPRECATED_START
template <typename Type, typename Value>
typename std::enable_if<
    ngraph::HasTypeInfoMember<Type>::value &&
        std::is_convertible<decltype(std::declval<Value>()->get_type_info().is_castable(Type::type_info)), bool>::value,
    bool>::type
is_type(Value value) {
    return value->get_type_info().is_castable(Type::type_info);
}

template <typename Type, typename Value>
typename std::enable_if<
    !ngraph::HasTypeInfoMember<Type>::value &&
        std::is_convertible<decltype(std::declval<Value>()->get_type_info().is_castable(Type::get_type_info_static())),
                            bool>::value,
    bool>::type
is_type(Value value) {
    return value->get_type_info().is_castable(Type::get_type_info_static());
}
OPENVINO_SUPPRESS_DEPRECATED_END

/// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
template <typename Type, typename Value>
typename std::enable_if<std::is_convertible<decltype(static_cast<Type*>(std::declval<Value>())), Type*>::value,
                        Type*>::type
as_type(Value value) {
    return ov::is_type<Type>(value) ? static_cast<Type*>(value) : nullptr;
}

namespace util {
template <typename T>
struct AsTypePtr;
/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename In>
struct AsTypePtr<std::shared_ptr<In>> {
    template <typename Type>
    static std::shared_ptr<Type> call(const std::shared_ptr<In>& value) {
        return ov::is_type<Type>(value) ? std::static_pointer_cast<Type>(value) : std::shared_ptr<Type>();
    }
};
}  // namespace util

/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename T, typename U>
auto as_type_ptr(const U& value) -> decltype(::ov::util::AsTypePtr<U>::template call<T>(value)) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return ::ov::util::AsTypePtr<U>::template call<T>(value);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
}  // namespace ov

namespace std {
template <>
struct OPENVINO_API hash<ov::DiscreteTypeInfo> {
    size_t operator()(const ov::DiscreteTypeInfo& k) const;
};
}  // namespace std
