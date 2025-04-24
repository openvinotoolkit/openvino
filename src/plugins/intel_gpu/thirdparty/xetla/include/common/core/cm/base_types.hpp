/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/core/cm/common.hpp"
#else
#include "common/core/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_base_types
/// @{

/// @brief xetla bf16 data type.
/// The difference between bf16 and fp32 is:
///
/// fp32: 0_00000000_00000000000000000000000
///
/// bf16: 0_00000000_0000000
/// @note
/// The member function in bf16 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct bf16 {
    uint16_t data;

    operator float() const {
        uint32_t temp = data;
        temp = temp << 0x10;
        return *reinterpret_cast<float *>(&temp);
    }
    bf16() = default;
    bf16(float val) { data = (*reinterpret_cast<uint32_t *>(&val)) >> 0x10; }

    bf16 &operator=(float val) {
        this->data = (*reinterpret_cast<uint32_t *>(&val)) >> 0x10;
        return *this;
    }
};

using fp16 = half;

/// @brief xetla tf32 data type.
/// The difference between tf32 and fp32 is:
///
/// fp32: 0_00000000_00000000000000000000000
///
/// tf32: 0_00000000_0000000000
/// @note
/// The member function in tf32 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct tf32 {
    uint32_t data;
    operator float() const {
        uint32_t temp = data;
        return *reinterpret_cast<float *>(&temp);
    }

    tf32(float val) {
        data = (*reinterpret_cast<uint32_t *>(&val)) & 0xFFFFE000;
    }

    tf32 &operator=(float val) {
        this->data = (*reinterpret_cast<uint32_t *>(&val)) & 0xFFFFE000;
        return *this;
    }
};

template <typename T>
struct get_packed_num {
    static constexpr uint32_t value = 1;
};

/// @brief Used to check if the type is xetla internal data type
/// @tparam T is the data type
template <typename T>
struct is_internal_type {
    static constexpr bool value = std::is_same<remove_const_t<T>, bf16>::value
            || std::is_same<remove_const_t<T>, tf32>::value;
};

/// @brief Used to check if the type is floating_point.
/// @tparam T is the data type
template <typename T>
struct is_floating_point {
    static constexpr bool value = std::is_same<remove_const_t<T>, bf16>::value
            || std::is_same<remove_const_t<T>, fp16>::value
            || std::is_same<remove_const_t<T>, tf32>::value
            || std::is_same<remove_const_t<T>, float>::value
            || std::is_same<remove_const_t<T>, double>::value;
};

/// @brief Used to check if the type is floating_point.
/// @tparam T is the data type
template <typename T>
struct is_integral {
    static constexpr bool value = std::is_same<remove_const_t<T>, int8_t>::value
            || std::is_same<remove_const_t<T>, uint8_t>::value
            || std::is_same<remove_const_t<T>, int16_t>::value
            || std::is_same<remove_const_t<T>, uint16_t>::value
            || std::is_same<remove_const_t<T>, int32_t>::value
            || std::is_same<remove_const_t<T>, uint32_t>::value
            || std::is_same<remove_const_t<T>, int64_t>::value
            || std::is_same<remove_const_t<T>, uint64_t>::value;
};

/// @brief Set the native data type of T
/// @tparam T is the data type
template <typename T>
struct native_type {
    using type = T;
};

/// @brief Set bfloat16 as the native data type of bf16
template <>
struct native_type<bf16> {
    using type = short;
};

/// @brief Set uint32_t as the native data type of tf32
template <>
struct native_type<tf32> {
    using type = int;
};

/// @brief Return the native data type of T
template <typename T>
using native_type_t = typename native_type<T>::type;

/// @brief Get the unit representation of type T
template <typename T>
struct uint_type {
    static constexpr bool is_uint8 = sizeof(T) == 1;
    static constexpr bool is_uint16 = sizeof(T) == 2;
    static constexpr bool is_uint32 = sizeof(T) == 4;
    static constexpr bool is_uint64 = sizeof(T) == 8;
    using type = typename std::conditional<is_uint8, uint8_t,
            typename std::conditional<is_uint16, uint16_t,
                    typename std::conditional<is_uint32, uint32_t,
                            typename std::conditional<is_uint64, uint64_t,
                                    void>::type>::type>::type>::type;
};

/// @brief Get the unit representation based on Size
template <int Size>
struct get_uint_type {
    static constexpr bool is_uint8 = Size == 1;
    static constexpr bool is_uint16 = Size == 2;
    static constexpr bool is_uint32 = Size == 4;
    static constexpr bool is_uint64 = Size == 8;
    using type = typename std::conditional<is_uint8, uint8_t,
            typename std::conditional<is_uint16, uint16_t,
                    typename std::conditional<is_uint32, uint32_t,
                            typename std::conditional<is_uint64, uint64_t,
                                    void>::type>::type>::type>::type;
};
/// @brief Return the uint representation of type T
template <typename T>
using uint_type_t = typename uint_type<T>::type;

/// @brief Return the uint representation based on Size
template <int Size>
using get_uint_type_t = typename get_uint_type<Size>::type;

/// @brief wrapper for xetla_vector.
/// Alias to CM `vector`.
/// @tparam Ty data type in xetla_vector.
/// @tparam N  data length in xetla_vector.
///
template <typename Ty, uint32_t N>
using xetla_vector = vector<native_type_t<Ty>, N>;

/// @brief wrapper for xetla_mask.
/// Alias to CM `vector<uint16_t, N>`.
/// @tparam N  data length in xetla_mask.
///
template <uint32_t N>
using xetla_mask = vector<uint16_t, N>;

/// @brief wrapper for xetla_mask_int.
/// Alias to CM integer as packed mask.
/// @tparam N  number of bits as mask in xetla_mask_int.
///
template <uint32_t N>
using xetla_mask_int = uint32_t;

/// Alias to `(empty)` if go with CM path.
/// @see gpu::xetla::core::xetla_matrix_ref gpu::xetla::core::xetla_vector_ref
#define __REF__

/// @brief wrapper for xetla_vector_ref.
/// Alias to `vector_ref` if go with CM path.
/// @note Need to be used together with `__REF__`, i.e. `"xetla_vector_ref __REF__"` is the full declaration of xetla vector reference.
/// @tparam Ty data type in xetla_vector.
/// @tparam N  data length in xetla_vector.
///
template <typename Ty, uint32_t N>
using xetla_vector_ref = vector_ref<native_type_t<Ty>, N>;

///
/// @brief Description of nd tensor descriptor for load and store.
///
using xetla_tdescriptor = xetla_vector<uint32_t, 16>;

/// @brief Alias to xetla_vector<uint32_t, 16> reference.
#define xetla_tdescriptor_ref xetla_vector_ref<uint32_t, 16> __REF__

/// @brief wrapper for xetla_matrix_ref.
/// Alias to `matrix_ref` if go with CM path.
/// @note Need to be used together with `__REF__`, i.e. `"xetla_matrix_ref __REF__"` is the full declaration of xetla matrix reference.
/// @tparam Ty data type in xetla_matrix_ref.
/// @tparam N1 row num in xetla_matrix_ref.
/// @tparam N2 col num in xetla_matrix_ref.
///
template <typename Ty, uint32_t N1, uint32_t N2>
using xetla_matrix_ref = matrix_ref<native_type_t<Ty>, N1, N2>;

/// @} xetla_core_base_types

} // namespace gpu::xetla
