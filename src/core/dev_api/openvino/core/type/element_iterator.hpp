// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/core/type/nf4.hpp"

namespace ov {
namespace util {

/**
 * @brief Make bit mask by setting N less significant bits.
 *
 * @tparam T Type of value.
 * @param n  Number of bits to set.
 * @return   Bit-mask value with N bits set.
 */
template <class T>
constexpr T make_n_bit_mask(const T n) {
    return (1ULL << n) - 1ULL;
}
}  // namespace util

namespace element {

/**
 * @brief Checks if element type is N in-raw bits type.
 *
 * @param et  Element type to check
 * @return True if element type is bit type otherwise false.
 */
constexpr bool is_bit_type(Type_t et) {
    return et == u1 || et == u2;
}

/**
 * @brief Checks if element type is 4-bits type.
 *
 * @param et  Element type to check
 * @return True if element type is nibble type otherwise false.
 */
constexpr bool is_nibble_type(Type_t et) {
    return et == u4 || et == i4 || et == nf4 || et == f4e2m1;
}

/**
 * @brief Checks if element type is split bit type.
 *
 * The value is stored in byte(s) like [b0, b1, x, .., x, b2, b3].
 *
 * @param et  Element type to check
 * @return True if element type is split bit type otherwise false.
 */
constexpr bool is_split_bit_type(Type_t et) {
    return et == u3 || et == u6;
}

/**
 * @brief Checks element type is using only N bytes as value.
 *
 * @param et  Element type to check.
 * @return True if element type use byte(s) for its value, false otherwise.
 */
constexpr bool is_byte_type(Type_t et) {
    return !is_bit_type(et) && !is_split_bit_type(et) && !is_nibble_type(et) && et != string;
}

/**
 * @brief Gets bit width of ov::element::Type_t.
 *
 * @return Number of bits representing the Type_t.
 */
template <Type_t ET>
constexpr size_t bit_width() {
    return sizeof(typename ov::fundamental_type_for<ET>());
}

template <>
constexpr size_t bit_width<Type_t::u1>() {
    return 1;
}

template <>
constexpr size_t bit_width<Type_t::u2>() {
    return 2;
}

template <>
constexpr size_t bit_width<Type_t::u3>() {
    return 3;
}

template <>
constexpr size_t bit_width<Type_t::u4>() {
    return 4;
}

template <>
constexpr size_t bit_width<Type_t::nf4>() {
    return 4;
}

template <>
constexpr size_t bit_width<Type_t::i4>() {
    return 4;
}

template <>
constexpr size_t bit_width<Type_t::u6>() {
    return 6;
}

template <>
constexpr size_t bit_width<Type_t::f4e2m1>() {
    return 4;
}

/**
 * @brief The BitProxy value class used by ov::element::Iterator to access values which has no standard byte(s) layout.
 *
 * It used by iterator to access values represented by precisions like u2, i4, u6 etc. in the way like stored
 * on bytes.
 * The R/W access is done via conversion and copy assignment operators.
 * The public members are used to work on sub-byte value like on its fundamental type defined by T.
 *
 * @tparam T       Fundamental type of sub-byte value which must be same as fundamental type of element::Type_t.
 * @tparam ET      OpenVINO element type.
 * @tparam Enable  Type to enable/disable this class.
 */
template <class T, Type_t ET, class Enable = void>
class BitProxy {};

/**
 * @brief The BitProxy specialization for types which are represented by N in-raw bits in byte.
 *
 * @tparam T  Fundamental type of sub-byte value which must be same as fundamental type of element::Type_t.
 * @tparam ET OpenVINO element type.
 */
template <class T, Type_t ET>
class BitProxy<T, ET, typename std::enable_if<is_bit_type(ET) || is_nibble_type(ET)>::type> {
private:
    template <Type_t, class>
    friend class Iterator;  //!< Iterator class is friend to access private members to manipulate pointer.

    using Bits = typename std::conditional<std::is_const<T>::value, const uint8_t, uint8_t>::type;

    static constexpr size_t m_bits = bit_width<ET>();                            //!< Number of bit for single value.
    static constexpr size_t m_num_values = 8 / m_bits;                           //!< Number values in byte.
    static constexpr size_t m_shift_init = is_nibble_type(ET) ? 0 : 8 - m_bits;  //!< Initial value for bit shift.

    Bits* m_ptr;         //!< Pointer to T as Bits used to get value from bits.
    size_t m_bit_shift;  //!< Current bit shift to get value.

    constexpr BitProxy(T* ptr) noexcept : m_ptr{reinterpret_cast<Bits*>(ptr)}, m_bit_shift{m_shift_init} {}

    uint8_t get_bit_value() const {
        constexpr auto value_mask = util::make_n_bit_mask(m_bits);
        return (*m_ptr >> m_bit_shift) & value_mask;
    }

    void set_bit_value(uint8_t value) {
        constexpr auto value_mask = util::make_n_bit_mask(m_bits);
        *m_ptr &= ~(value_mask << m_bit_shift);
        *m_ptr |= value << m_bit_shift;
    }

public:
    using value_type = typename std::decay<T>::type;  //!< Fundamental type of bound to BitProxy.

    /**
     * @brief Compare proxy value with other provided value.
     * @param rhs  Value to compare.
     * @return True if equal otherwise false.
     */
    template <class U>
    constexpr bool operator==(const U& rhs) const {
        return static_cast<value_type>(*this) == rhs;
    }

    /**
     * @brief Compare proxy value is less than rhs.
     *
     * @tparam U   Type of value to compare.
     * @param rhs  Value to compare.
     * @return True if less otherwise false.
     */
    template <class U>
    constexpr bool operator<(const U& rhs) const {
        return static_cast<value_type>(*this) < rhs;
    }

    /**
     * @brief Converts to fundamental type.
     *
     * @return Value of BitProxy.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT != i4 && ETT != f4e2m1>::type* = nullptr>
    operator value_type() const {
        return static_cast<value_type>(get_bit_value());
    }

    /**
     * @brief Converts to fundamental type.
     *
     * @return Value of BitProxy.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT == f4e2m1>::type* = nullptr>
    operator value_type() const {
        return value_type::from_bits(get_bit_value());
    }

    /**
     * @brief Converts to float for NF4.
     *
     * @note Implementation aligned to ConvertNF4::unpack, de-quantization applied only when converting to
     * floating point.  For integral types get bit value.
     *
     * @return Converted NF4 value to float.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT == nf4>::type* = nullptr>
    operator float() const {
        return ConvertNF4::dequantize(get_bit_value());
    }

    template <Type_t ETT = ET, typename std::enable_if<ETT == f4e2m1>::type* = nullptr>
    operator float() const {
        return static_cast<float>(float4_e2m1::from_bits(get_bit_value()));
    }

    /**
     * @brief Converts to fundamental type.
     *
     * @return Value of BitProxy.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT == i4>::type* = nullptr>
    operator value_type() const {
        constexpr auto value_mask = util::make_n_bit_mask(m_bits);
        constexpr auto value_msb_mask = (1U << (m_bits - 1U));

        auto v = get_bit_value();
        if (v & value_msb_mask) {
            // If N bit value MSB bit is set then value is negative.
            // As v is byte then all bits above N must be set to be two's complement.
            v |= ~value_mask;
        }
        return static_cast<value_type>(v);
    }

    /**
     * @brief Sets current ProxyBit to value.
     * @param v  Value to be set.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT != f4e2m1>::type* = nullptr>
    BitProxy<T, ET>& operator=(const value_type v) {
        constexpr auto value_mask = util::make_n_bit_mask(m_bits);
        set_bit_value(static_cast<uint8_t>(v) & value_mask);
        return *this;
    }

    /**
     * @brief Sets current ProxyBit to value (f4e2m1 specialization).
     * @param v  Value to be set.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT == f4e2m1>::type* = nullptr>
    BitProxy<T, ET>& operator=(const value_type v) {
        set_bit_value(v.to_bits());
        return *this;
    }

    /**
     * @brief Sets current NF4 value from float using quantization.
     * @param v  Value to be set.
     */
    template <Type_t ETT = ET, typename std::enable_if<ETT == nf4>::type* = nullptr>
    BitProxy<T, ET>& operator=(const float v) {
        set_bit_value(ConvertNF4::quantize(v));
        return *this;
    }
};

/**
 * @brief The BitProxy specialization for u3, u6 precisions.
 *
 * @note The input pointer must point on buffer which has got 3 * n bytes.
 *
 * @tparam T  Fundamental type of sub-byte value which must be same as fundamental type of element::Type_t.
 * @tparam ET OpenVINO element type.
 */
template <class T, Type_t ET>
class BitProxy<T, ET, typename std::enable_if<is_split_bit_type(ET)>::type> {
private:
    template <Type_t, class>
    friend class Iterator;  //!< Iterator class is friend to access private members to manipulate pointer.

    static constexpr size_t m_bits = bit_width<ET>();         //!< Number of bit for single value.
    static constexpr size_t m_num_values = (3 * 8) / m_bits;  //!< Number values in byte.
    static constexpr size_t m_shift_init = m_num_values - 1;  //!< Initial value for bit shift.

    struct ByteValue {
        uint8_t b0;
        uint8_t b1;
        uint8_t b2;
    };

    union {
        T* m_ptr;            //!< Pointer to T buffer.
        ByteValue* m_bytes;  //!< Pointer to buffer as 3 bytes representation.
    };

    size_t m_bit_shift;  //!< Current bit shift to get value.

    constexpr BitProxy(T* ptr) noexcept : m_ptr{ptr}, m_bit_shift{m_shift_init} {}

public:
    using value_type = typename std::decay<T>::type;  //!< Fundamental type of sub-byte.

    /**
     * @brief Compare proxy value is equal than rhs.
     *
     * @tparam U   Type of value to compare.
     * @param rhs  Value to compare.
     * @return True if equal, false otherwise.
     */
    template <class U>
    constexpr bool operator==(const U& rhs) const {
        return static_cast<value_type>(*this) == rhs;
    }

    /**
     * @brief Compare proxy value is less than rhs.
     *
     * @tparam U   Type of value to compare.
     * @param rhs  Value to compare.
     * @return True if less otherwise false.
     */
    template <class U>
    constexpr bool operator<(const U& rhs) const {
        return static_cast<value_type>(*this) < rhs;
    }

    /**
     * @brief Converts to fundamental type.
     *
     * @return Value of BitProxy.
     */
    operator value_type() const {
        constexpr uint16_t lower_mask_bits = 16 / m_num_values;
        constexpr uint16_t upper_mask_bits = 8 / m_num_values;
        constexpr uint16_t mask_lower = util::make_n_bit_mask(lower_mask_bits);
        constexpr uint16_t mask_upper = util::make_n_bit_mask(upper_mask_bits) << lower_mask_bits;

        // get lower part of value
        uint16_t v = ((m_bytes->b0 << 8U) | m_bytes->b1) >> (lower_mask_bits * m_bit_shift);
        v &= mask_lower;
        // get upper part of value
        v |= ((m_bytes->b2 << lower_mask_bits) >> (upper_mask_bits * m_bit_shift)) & mask_upper;
        return static_cast<value_type>(v);
    }

    /**
     * @brief Sets current ProxyBit to value.
     * @param v  Value to be set.
     */
    BitProxy<T, ET>& operator=(const value_type v) {
        constexpr uint16_t lower_mask_bits = 16 / m_num_values;
        constexpr uint16_t upper_mask_bits = 8 / m_num_values;
        constexpr uint16_t mask_lower = util::make_n_bit_mask(lower_mask_bits);
        constexpr uint16_t mask_upper = util::make_n_bit_mask(upper_mask_bits) << lower_mask_bits;

        uint16_t tmp = (m_bytes->b0 << 8U) | m_bytes->b1;
        tmp &= ~(mask_lower << (lower_mask_bits * m_bit_shift));
        tmp |= (v & mask_lower) << (lower_mask_bits * m_bit_shift);
        m_bytes->b0 = tmp >> 8U;
        m_bytes->b1 = tmp & 0x00ff;

        tmp = m_bytes->b2 & ~((mask_upper >> lower_mask_bits) << (upper_mask_bits * m_bit_shift));
        tmp |= (((v & mask_upper) >> lower_mask_bits) << (upper_mask_bits * m_bit_shift));
        m_bytes->b2 = tmp & 0x00ff;
        return *this;
    }
};

/**
 * @brief Put BitProxy value to output stream.
 *
 * @param os    Reference to output stream.
 * @param value Value to print.
 * @return return output stream.
 */
template <class T, Type_t ET>
std::ostream& operator<<(std::ostream& os, const BitProxy<T, ET>& value) {
    os << +static_cast<T>(value);
    return os;
}

/**
 * @brief Bidirectional iterator of specified precision.
 *
 * The iterator supports low precisions using BitProxy to access values via conversion.
 *
 * @tparam ET  Type of OpenVINO element type (ov::element::Type_t).
 * @tparam T   Must be fundamental type for specified ET.
 */
template <Type_t ET, class T>
class Iterator {
    using proxy_type = BitProxy<T, ET>;

public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = typename std::conditional<std::is_const<T>::value, const proxy_type&, proxy_type&>::type;
    using pointer = typename std::conditional<std::is_const<T>::value, const proxy_type*, proxy_type*>::type;

    static_assert(std::is_same<typename std::decay<T>::type, ov::fundamental_type_for<ET>>::value,
                  "Iterator value_type must be same as fundamental type of ET");

    constexpr Iterator(T* ptr) noexcept : m_et_ptr{ptr} {}

    template <class U>
    constexpr Iterator(U* ptr) noexcept : m_et_ptr{reinterpret_cast<T*>(ptr)} {
        static_assert(std::is_same<typename std::decay<U>::type, int8_t>::value,
                      "User type must be int8_t as base type for LP ET");
    }

    // Iteration operators
    template <Type_t ETT = ET>
    typename std::enable_if<is_bit_type(ETT), Iterator<ET, T>>::type& operator++() {
        m_et_ptr.m_bit_shift -= m_et_ptr.m_bits;
        m_et_ptr.m_bit_shift = m_et_ptr.m_bit_shift % (m_et_ptr.m_num_values * m_et_ptr.m_bits);
        m_et_ptr.m_ptr += static_cast<std::ptrdiff_t>(m_et_ptr.m_bit_shift == m_et_ptr.m_shift_init);
        return *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_nibble_type(ETT), Iterator<ET, T>>::type& operator++() {
        m_et_ptr.m_bit_shift ^= m_et_ptr.m_bits;
        m_et_ptr.m_ptr += static_cast<std::ptrdiff_t>(m_et_ptr.m_bit_shift == m_et_ptr.m_shift_init);
        return *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_split_bit_type(ETT), Iterator<ET, T>>::type& operator++() {
        --m_et_ptr.m_bit_shift;
        m_et_ptr.m_bit_shift = m_et_ptr.m_bit_shift % m_et_ptr.m_num_values;
        m_et_ptr.m_ptr += (m_et_ptr.m_bit_shift == m_et_ptr.m_shift_init) ? 3 : 0;
        return *this;
    }

    Iterator<ET, T> operator++(int) {
        auto old = *this;
        ++(*this);
        return old;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_bit_type(ETT), Iterator<ET, T>>::type& operator+=(const difference_type& n) {
        const auto advance = n + (m_et_ptr.m_shift_init - m_et_ptr.m_bit_shift) / m_et_ptr.m_bits;
        m_et_ptr.m_bit_shift = m_et_ptr.m_shift_init - (advance % m_et_ptr.m_num_values) * m_et_ptr.m_bits;
        m_et_ptr.m_ptr += advance / m_et_ptr.m_num_values;
        return *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_nibble_type(ETT), Iterator<ET, T>>::type& operator+=(const difference_type& n) {
        m_et_ptr.m_ptr += n / m_et_ptr.m_num_values;
        return (n % m_et_ptr.m_num_values) ? ++*this : *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_split_bit_type(ETT), Iterator<ET, T>>::type& operator+=(const difference_type& n) {
        const auto advance = n + m_et_ptr.m_shift_init - m_et_ptr.m_bit_shift;
        m_et_ptr.m_bit_shift = m_et_ptr.m_shift_init - (advance % m_et_ptr.m_num_values);
        m_et_ptr.m_ptr += 3 * (advance / m_et_ptr.m_num_values);
        return *this;
    }

    Iterator<ET, T> operator+(const difference_type& n) const {
        auto tmp(*this);
        tmp += n;
        return tmp;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_bit_type(ETT), Iterator<ET, T>>::type& operator--() {
        m_et_ptr.m_bit_shift += m_et_ptr.m_bits;
        m_et_ptr.m_bit_shift = m_et_ptr.m_bit_shift % (m_et_ptr.m_num_values * m_et_ptr.m_bits);
        m_et_ptr.m_ptr -= static_cast<std::ptrdiff_t>(m_et_ptr.m_bit_shift == 0);
        return *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_nibble_type(ETT), Iterator<ET, T>>::type& operator--() {
        m_et_ptr.m_bit_shift ^= m_et_ptr.m_bits;
        m_et_ptr.m_ptr -= static_cast<std::ptrdiff_t>(m_et_ptr.m_bit_shift == 4);
        return *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_split_bit_type(ETT), Iterator<ET, T>>::type& operator--() {
        ++m_et_ptr.m_bit_shift;
        m_et_ptr.m_bit_shift = m_et_ptr.m_bit_shift % m_et_ptr.m_num_values;
        m_et_ptr.m_ptr -= m_et_ptr.m_bit_shift == 0 ? 3 : 0;
        return *this;
    }

    Iterator<ET, T> operator--(int) {
        auto old = *this;
        --(*this);
        return old;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_bit_type(ETT), Iterator<ET, T>>::type& operator-=(const difference_type& n) {
        const auto advance = m_et_ptr.m_bit_shift / m_et_ptr.m_bits + n;
        m_et_ptr.m_bit_shift = (advance % m_et_ptr.m_num_values) * m_et_ptr.m_bits;
        m_et_ptr.m_ptr -= advance / m_et_ptr.m_num_values;
        return *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_nibble_type(ETT), Iterator<ET, T>>::type& operator-=(const difference_type& n) {
        m_et_ptr.m_ptr -= n / m_et_ptr.m_num_values;
        return (n % m_et_ptr.m_num_values) ? --*this : *this;
    }

    template <Type_t ETT = ET>
    typename std::enable_if<is_split_bit_type(ETT), Iterator<ET, T>>::type& operator-=(const difference_type& n) {
        const auto advance = m_et_ptr.m_bit_shift + n;
        m_et_ptr.m_bit_shift = advance % m_et_ptr.m_num_values;
        m_et_ptr.m_ptr -= 3 * (advance / m_et_ptr.m_num_values);
        return *this;
    }

    Iterator<ET, T> operator-(const difference_type& n) const {
        auto tmp(*this);
        tmp -= n;
        return tmp;
    }

    // compare operators
    constexpr bool operator!=(const Iterator<ET, T>& rhs) const {
        return (m_et_ptr.m_ptr != rhs.m_et_ptr.m_ptr) || (m_et_ptr.m_bit_shift != rhs.m_et_ptr.m_bit_shift);
    }

    // dereference operators
    constexpr const proxy_type& operator*() const {
        return m_et_ptr;
    }

    reference operator*() {
        return m_et_ptr;
    }

private:
    proxy_type m_et_ptr;
};

/**
 * @brief Make element iterator from pointer.
 *
 * @tparam ET  Type of ov::element::Type_t.
 * @tparam T   Type of pointer data. Must be fundamental type of ET.
 *
 * @param ptr  Pointer to data.
 * @return Element iterator for type ET.
 */
template <Type_t ET, class T, typename std::enable_if<!is_byte_type(ET) && ET != string>::type* = nullptr>
constexpr Iterator<ET, ov::fundamental_type_for<ET>> iterator(T* ptr) {
    return {ptr};
}

/**
 * @brief Make element iterator from pointer.
 *
 * @tparam ET  Type of ov::element::Type_t.
 * @tparam T   Type of pointer data. Must be fundamental type of ET.
 *
 * @param ptr  Pointer to data.
 * @return Element iterator for type ET.
 */
template <Type_t ET, class T, typename std::enable_if<!is_byte_type(ET) && ET != string>::type* = nullptr>
constexpr Iterator<ET, const ov::fundamental_type_for<ET>> iterator(const T* ptr) {
    return {ptr};
}

/**
 * @brief Make iterator from pointer for standard types.
 *
 * To have common interface for all ov::element::Type. Just return input pointer.
 *
 * @tparam ET  Type of ov::element::Type_t.
 * @tparam T   Type of pointer data. Must be fundamental type of ET.
 *
 * @param ptr  Pointer to data.
 * @return Element iterator same as input pointer.
 */
template <Type_t ET, class T, typename std::enable_if<is_byte_type(ET) || ET == string>::type* = nullptr>
constexpr T* iterator(T* ptr) {
    return ptr;
}

/**
 * @brief Make iterator from void pointer.
 *
 * Data will be reinterpreted using fundamental type for ov::element::Type.
 *
 * @tparam ET  OpenVINO element type.
 * @param ptr  Pointer to data.
 * @return Iterator for given ET.
 */
template <Type_t ET, class T = ov::fundamental_type_for<ET>>
constexpr auto iterator(void* ptr) -> decltype(iterator<ET, T>(reinterpret_cast<T*>(ptr))) {
    return iterator<ET, T>(reinterpret_cast<T*>(ptr));
}

/**
 * @brief Make iterator from constant void pointer.
 *
 * Data will be reinterpreted using fundamental type for ov::element::Type.
 *
 * @tparam ET  OpenVINO element type.
 * @param ptr  Pointer to data.
 * @return Iterator for given ET.
 */
template <Type_t ET, class T = typename std::add_const<ov::fundamental_type_for<ET>>::type>
constexpr auto iterator(const void* ptr) -> decltype(iterator<ET, T>(reinterpret_cast<T*>(ptr))) {
    return iterator<ET, T>(reinterpret_cast<T*>(ptr));
}

/**
 * @brief Gets size of memory in bytes for N elements of given precision.
 *
 * @param type  Element precision.
 * @param n     Number of elements.
 *
 * @return Elements size in bytes.
 */
OPENVINO_API size_t get_memory_size(const element::Type& type, const size_t n);
}  // namespace element
}  // namespace ov
