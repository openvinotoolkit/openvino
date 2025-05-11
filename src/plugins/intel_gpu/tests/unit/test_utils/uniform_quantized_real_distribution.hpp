// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tests { namespace distributions {

/// @cond PRIVATE
namespace detail
{

/// @brief Rounds the number down (floor) to the nearest lower-equal multiply of specified @p base.
///
/// @param val  Rounded value.
/// @param base Base which is used for rounding.
/// @return     Number lower-equal to @p val that is multiply of @p base.
static constexpr std::size_t round_down_to_multiply(const std::size_t val, const std::size_t base)
{
    return val / base * base;
}

/// @brief Returns result of division rounded up to nearest greater-equal integer.
///
/// @param val     Divided value.
/// @param divider Divider.
/// @return        Ceiling of division result.
static constexpr std::size_t div_ceil(const std::size_t val, const std::size_t divider)
{
    return (val + divider - 1U) / divider;
}

/// @brief Rounds the number up (ceiling) to the nearest greater-equal multiply of specified @p base.
///
/// @param val  Rounded value.
/// @param base Base which is used for rounding.
/// @return     Number greater-equal to @p val that is multiply of @p base.
static constexpr std::size_t round_up_to_multiply(const std::size_t val, const std::size_t base)
{
    return div_ceil(val, base) * base;
}

// --------------------------------------------------------------------------------------------------------------------

static constexpr std::size_t log2_floor_helper1_(const std::size_t val, const std::size_t log)
{
    return val <= 1U ? log : log2_floor_helper1_(val >> 1U, log + 1U);
}

/// @brief Calculates log (base 2) from specified @p val. Results is trunced (floor) to integer.
///
/// @details For @c 0, it returns @c 0.
///
/// @param val Value which <tt>floor(log_2(.))</tt> will be calculated.
/// @return    Unsigned integer @a x for which <tt>2^x</tt> is greatest value lower-equal to @p val.
static constexpr std::size_t log2_floor(const std::size_t val)
{
    return log2_floor_helper1_(val, 0U);
}

// --------------------------------------------------------------------------------------------------------------------

template <typename DataType,
          bool IsFloatingPoint = (std::is_floating_point<DataType>::value &&
              std::numeric_limits<DataType>::is_iec559 &&
              std::numeric_limits<DataType>::radix == 2 &&
              std::numeric_limits<DataType>::digits > 0),
          bool IsIntegral = (std::is_integral<DataType>::value &&
              std::numeric_limits<DataType>::is_integer &&
              std::numeric_limits<DataType>::radix == 2 &&
              std::numeric_limits<DataType>::digits > 0)>
struct estimate_repr_size_helper2_
{
    /// @brief Estimated size of representation in bits.
    static constexpr std::size_t value = sizeof(DataType) * CHAR_BIT;
};

template <typename DataType, bool IsIntegral>
struct estimate_repr_size_helper2_<DataType, true, IsIntegral>
{
private:
    /// @brief Estimated number of values needed to represent exponent of floating point number.
    static constexpr std::size_t exponent_vals_cnt = (std::numeric_limits<DataType>::max_exponent -
            std::numeric_limits<DataType>::min_exponent + 1U) +
        static_cast<std::size_t>(std::numeric_limits<DataType>::has_infinity ||
            std::numeric_limits<DataType>::has_quiet_NaN ||
            std::numeric_limits<DataType>::has_signaling_NaN) +
        static_cast<std::size_t>(std::numeric_limits<DataType>::has_denorm != std::float_denorm_style::denorm_absent);

public:
    /// @brief Estimated size of representation in bits.
    static constexpr std::size_t value = round_up_to_multiply(
        static_cast<std::size_t>(std::numeric_limits<DataType>::is_signed) +
        log2_floor(exponent_vals_cnt) +
        (std::numeric_limits<DataType>::digits - 1U), CHAR_BIT);
};

template <typename DataType>
struct estimate_repr_size_helper2_<DataType, false, true>
{
    /// @brief Estimated size of representation in bits.
    static constexpr std::size_t value = round_up_to_multiply(
        static_cast<std::size_t>(std::numeric_limits<DataType>::is_signed) +
        std::numeric_limits<DataType>::digits, CHAR_BIT);
};

template <typename DataType,
          bool IsArithmetic = std::is_arithmetic<DataType>::value>
struct estimate_repr_size_helper1_
{
    static_assert(std::is_object<DataType>::value, "DataType must be an object type.");

    /// @brief Estimated size of representation in bits.
    static constexpr std::size_t value = sizeof(DataType) * CHAR_BIT;
};

template <typename DataType>
struct estimate_repr_size_helper1_<DataType, true>
    : estimate_repr_size_helper2_<DataType> {};

/// @brief Estimates size in bits (rounded to byte) of representation of specified DataType.
///
/// @tparam DataType Data type which size of representation will be estimated.
template <typename DataType>
struct estimate_repr_size : estimate_repr_size_helper1_<DataType> {};

// --------------------------------------------------------------------------------------------------------------------

/// @brief Guard that saves / restores upon destruction format state of stream.
///
/// @tparam CharType   Character type use in stream.
/// @tparam CharTraits Traits describing character set / stream behavior.
template <typename CharType, typename CharTraits>
struct ios_format_guard
{
    // ----------------------------------------------------------------------------------------------------------------
    // Typedefs, aliases and constants (including nested types and their helpers).
    // ----------------------------------------------------------------------------------------------------------------
    using char_type = CharType;
    using char_traits = CharTraits;

    using stream_ios_type   = std::basic_ios<char_type, char_traits>;
    using format_flags_type = std::ios_base::fmtflags;
    using precision_type    = std::streamsize;

    // ----------------------------------------------------------------------------------------------------------------
    // Constructors, special functions, destructors.
    // ----------------------------------------------------------------------------------------------------------------
private:
    /// @brief Creates guard that saves and restores (during destruction) the stream properties / state connected
    ///        to format.
    ///
    /// @param stream_ios I/O state object of stream. Its lifetime must be wider than lifetime of guard.
    // ReSharper disable once CppDoxygenUndocumentedParameter
    ios_format_guard(stream_ios_type& stream_ios, int = 0)
        : _stream_ios(stream_ios),
          _old_format_flags(stream_ios.flags()),
          _old_precision(stream_ios.precision()),
          _old_fill(stream_ios.fill()) {}

public:
    /// @brief Creates guard that saves and restores (during destruction) the stream properties / state connected
    ///        to format.
    ///
    /// @param stream Input stream object. Its lifetime must be wider than lifetime of guard.
    // ReSharper disable once CppPossiblyUninitializedMember
    explicit ios_format_guard(std::basic_istream<CharType, CharTraits>& stream)
        : ios_format_guard(stream, 0) {}

    /// @brief Creates guard that saves and restores (during destruction) the stream properties / state connected
    ///        to format.
    ///
    /// @param stream Output stream object. Its lifetime must be wider than lifetime of guard.
    // ReSharper disable once CppPossiblyUninitializedMember
    explicit ios_format_guard(std::basic_ostream<CharType, CharTraits>& stream)
        : ios_format_guard(stream, 0) {}

    // Explicitly deleted/defaulted special functions:
    ios_format_guard(const ios_format_guard& other) = delete;
    ios_format_guard(ios_format_guard&& other) noexcept = default;
    ios_format_guard& operator =(const ios_format_guard& other) = delete;
    ios_format_guard& operator =(ios_format_guard&& other) noexcept = delete;

    /// @brief Restores stream properies / format state.
    ~ios_format_guard()
    {
        _stream_ios.flags(_old_format_flags);
        _stream_ios.precision(_old_precision);
        _stream_ios.fill(_old_fill);
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Data members.
    // ----------------------------------------------------------------------------------------------------------------
private:
    stream_ios_type& _stream_ios;

    format_flags_type _old_format_flags;
    precision_type    _old_precision;
    char_type         _old_fill;
};

/// @brief Creates guard that saves / restores upon destruction format state of stream.
///
/// @tparam CharType   Character type use in stream.
/// @tparam CharTraits Traits describing character set / stream behavior.
/// @param stream Stream for which guard is created.
/// @return       Format guard. Upon destruction it will restore format of stream.
template <typename CharType, typename CharTraits>
ios_format_guard<CharType, CharTraits> create_format_guard(std::basic_istream<CharType, CharTraits>& stream)
{
    return ios_format_guard<CharType, CharTraits>(stream);
}

/// @brief Creates guard that saves / restores upon destruction format state of stream.
///
/// @tparam CharType   Character type use in stream.
/// @tparam CharTraits Traits describing character set / stream behavior.
/// @param stream Stream for which guard is created.
/// @return       Format guard. Upon destruction it will restore format of stream.
template <typename CharType, typename CharTraits>
ios_format_guard<CharType, CharTraits> create_format_guard(std::basic_ostream<CharType, CharTraits>& stream)
{
    return ios_format_guard<CharType, CharTraits>(stream);
}

// --------------------------------------------------------------------------------------------------------------------

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4814)
#endif

/// @brief Mask that can manipulate bits of specified type.
///
/// @tparam DataType                Data type for which mask is created.
/// @tparam ForceOppositeEndianness Force endianness oposite to native one (for testing purposes).
template <typename DataType, bool ForceOppositeEndianness = false>
class data_type_mask
{
    // ----------------------------------------------------------------------------------------------------------------
    // Typedefs, aliases and constants (including nested types and their helpers).
    // ----------------------------------------------------------------------------------------------------------------
#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) \
    || defined(__LITTLE_ENDIAN__) || defined(__ARMEL__) || defined(__THUMBEL__) || defined(__AARCH64EL__)      \
    || defined(__MIPSEL__) || defined(_IA64) || defined(__ia64) || defined(__ia64__) || defined(__IA64__)      \
    || defined(__itanium__) || defined(_M_IA64) || defined(__INTEL__) || defined(_X86_) || defined(_M_IX86)    \
    || defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(__amd64__) || defined(_M_X64)   \
    || defined(_WIN32)
    // TODO: Use check for endianess (C++20) instead of pre-processor predefined constants.
    /// @brief Indicates thet we have little-endian endianess. It is @c true for all x86 / x86_64 platforms.
    static constexpr bool little_endian = !ForceOppositeEndianness; // ^ (std::endian::native == std::endian::little);
#else
    /// @brief Indicates thet we have little-endian endianess. It is @c true for all x86 / x86_64 platforms.
    static constexpr bool little_endian = ForceOppositeEndianness;
#endif

    /// @brief Type of part used as base unit of mask.
    using part_type = unsigned long long;
public:
    /// @brief Type of data for which mask is created and to which mask will be applied.
    using data_type = DataType;

private:
    /// @brief Mask part with all bits cleared (set to @c 0).
    static constexpr part_type part_zeros = static_cast<part_type>(0);
    /// @brief Mask part with all bits set (set to @c 1).
    static constexpr part_type part_ones  = ~part_zeros;

    /// @brief Number of bits in representation of data_type (without padding).
    static constexpr std::size_t data_repr_bit_size    = estimate_repr_size<data_type>::value;
    /// @brief Number of bits in representation of part_type (part type is used to store mask).
    static constexpr std::size_t data_part_bit_size    = CHAR_BIT * sizeof(part_type);
    /// @brief Number of parts needed to store representation of data_type.
    static constexpr std::size_t data_repr_parts_count = div_ceil(data_repr_bit_size, data_part_bit_size);

    /// @brief Size of end padding in bits in mask (bits that do not represent data_type).
    static constexpr std::size_t padding_bit_size   = data_repr_parts_count * data_part_bit_size - data_repr_bit_size;
    /// @brief Number of bits to include in last part of a mask. If it is @c 0, all bits are included.
    static constexpr std::size_t last_part_bit_size = data_repr_bit_size % data_part_bit_size;
    /// @brief Inclusion mask used to filter last part.
    static constexpr part_type last_part_mask       = little_endian
                                                          ? ~(part_ones << last_part_bit_size)
                                                          : ~(part_ones >> last_part_bit_size);

    /// @brief Marker type used to select correct constexpr constructor for data_parts (that will
    ///        zero-initalize all members).
    struct data_parts_zero_mark {};
    /// @brief Marker used to select constexpr constructor that zeroes entire data_parts.
    static constexpr data_parts_zero_mark data_parts_zero{}; // "{}": WA for clang 3.6-3.9

    /// @brief Helper structure to store and fast bit-copy mask.
    ///
    /// @details Type is POD type.
    struct data_parts
    {
        // ------------------------------------------------------------------------------------------------------------
        // data_parts: Typedefs, aliases and constants (including nested types and their helpers).
        // ------------------------------------------------------------------------------------------------------------

        // Redeclarations of some constants due to intransitive nature of friends.
        /// @brief Number of parts needed to store representation of data_type.
        static constexpr auto data_repr_parts_count = data_type_mask::data_repr_parts_count;

        // ------------------------------------------------------------------------------------------------------------
        // data_parts: Constructors, special functions, destructors.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Default constructor. Leaves parts uninitialized.
        data_parts() = default;
        /// @brief Constructor that zero-initializes all parts (clears all bits).
        // ReSharper disable once CppNonExplicitConvertingConstructor
        constexpr data_parts(data_parts_zero_mark)
            : parts{} {}

        // ------------------------------------------------------------------------------------------------------------
        // data_parts: Properties, Accessors.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Gets part by index (ordered from high-to-low mask bits).
        ///
        /// @param index Index of the part (@c 0-based). Must be lower than data_repr_parts_count.
        /// @return      Value of part specified by index.
        constexpr part_type part_hilo_ord(const std::size_t index) const noexcept
        {
            return little_endian ? parts[data_repr_parts_count - 1 - index] : parts[index];
        }

        /// @brief Accesses part by index (ordered from high-to-low mask bits).
        ///
        /// @param index Index of the part (@c 0-based). Must be lower than data_repr_parts_count.
        /// @return      Value of part specified by index.
        part_type& part_hilo_ord(const std::size_t index) noexcept
        {
            return little_endian ? parts[data_repr_parts_count - 1 - index] : parts[index];
        }

        /// @brief Sets part by index (ordered from high-to-low mask bits).
        ///
        /// @param index Index of the part (@c 0-based). Must be lower than data_repr_parts_count.
        /// @param val   New value for selected part.
        void part_hilo_ord(const std::size_t index, const part_type val) noexcept
        {
            (little_endian ? parts[data_repr_parts_count - 1U - index] : parts[index]) = val;
        }

        /// @brief Gets part by index (ordered from low-to-high mask bits).
        ///
        /// @param index Index of the part (@c 0-based). Must be lower than data_repr_parts_count.
        /// @return      Value of part specified by index.
        constexpr part_type part_lohi_ord(const std::size_t index) const noexcept
        {
            return little_endian ? parts[index] : parts[data_repr_parts_count - 1U - index];
        }

        /// @brief Accesses part by index (ordered from low-to-high mask bits).
        ///
        /// @param index Index of the part (@c 0-based). Must be lower than data_repr_parts_count.
        /// @return      Value of part specified by index.
        part_type& part_lohi_ord(const std::size_t index) noexcept
        {
            return little_endian ? parts[index] : parts[data_repr_parts_count - 1U - index];
        }

        /// @brief Sets part by index (ordered from low-to-high mask bits).
        ///
        /// @param index Index of the part (@c 0-based). Must be lower than data_repr_parts_count.
        /// @param val   New value for selected part.
        void part_lohi_ord(const std::size_t index, const part_type val) noexcept
        {
            (little_endian ? parts[index] : parts[data_repr_parts_count - 1U - index]) = val;
        }

        // ------------------------------------------------------------------------------------------------------------
        // data_parts: Functions.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Sets bits starting from specified index (bits before index are cleared).
        ///
        /// @details Sets mask starting from specified index of bit. All bits starting from @p from are set (set to
        /// @c 1). Bits before @c from are cleared (set to @c 0).
        ///
        /// @tparam IsZeroPreinited Indicates that parts were zero-pre-initialized and full clearing can be avoided.
        /// @param from             Index (@c 0-based) indicating first bit to set.
        /// @return                 Current instance of data_parts.
        template <bool IsZeroPreinited = false>
        data_parts& set_bits(std::size_t from) noexcept
        {
            const std::size_t adj_from = little_endian ? from : padding_bit_size + from;

            const std::size_t from_part           = adj_from / data_part_bit_size;
            const std::size_t from_part_start_bit = adj_from % data_part_bit_size;
            const part_type from_part_bits        = part_ones << from_part_start_bit;

            if (!IsZeroPreinited)
            {
                for (std::size_t pi = 0; pi < std::min(from_part, data_repr_parts_count); ++pi)
                    part_lohi_ord(pi, part_zeros);
            }

            if (from_part < data_repr_parts_count)
                part_lohi_ord(from_part, from_part_bits);
            for (std::size_t pi = from_part + 1U; pi < data_repr_parts_count; ++pi)
                part_lohi_ord(pi, part_ones);
            if (last_part_bit_size > 0U) // constexpr
                parts[data_repr_parts_count - 1] &= last_part_mask;

            return *this;
        }

        // ------------------------------------------------------------------------------------------------------------
        // data_parts: Operators.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Negates all bits of the mask.
        ///
        /// @return Data parts with all bits negated.
        data_parts operator ~() const noexcept
        {
            data_parts result(data_parts_zero);
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                result.parts[pi] = ~parts[pi];
            if (last_part_bit_size > 0U) // constexpr
                result.parts[data_repr_parts_count - 1] &= last_part_mask;
            return result;
        }

        /// @brief Performs bitwise-AND of two data_parts objects.
        ///
        /// @param rhs Right-hand side.
        /// @return    Data parts with resulting bitwise-AND bits.
        data_parts operator &(const data_parts& rhs) const noexcept
        {
            data_parts result(data_parts_zero);
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                result.parts[pi] = parts[pi] & rhs.parts[pi];
            return result;
        }

        /// @brief Performs bitwise-OR of two data_parts objects.
        ///
        /// @param rhs Right-hand side.
        /// @return    Data parts with resulting bitwise-OR bits.
        data_parts operator |(const data_parts& rhs) const noexcept
        {
            data_parts result(data_parts_zero);
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                result.parts[pi] = parts[pi] | rhs.parts[pi];
            return result;
        }

        /// @brief Performs bitwise-XOR of two data_parts objects.
        ///
        /// @param rhs Right-hand side.
        /// @return    Data parts with resulting bitwise-XOR bits.
        data_parts operator ^(const data_parts& rhs) const noexcept
        {
            data_parts result(data_parts_zero);
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                result.parts[pi] = parts[pi] ^ rhs.parts[pi];
            return result;
        }

        /// @brief Performs bitwise-MINUS of two data_parts objects.
        /// @details Bitwise-MINUS is similar to difference of two sets, equivalent to:
        ///          <tt>A - B === A & ~B</tt>.
        ///
        /// @param rhs Right-hand side.
        /// @return    Data parts with resulting bitwise-MINUS bits.
        data_parts operator -(const data_parts& rhs) const noexcept
        {
            data_parts result(data_parts_zero);
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                result.parts[pi] = parts[pi] & ~rhs.parts[pi];
            if (last_part_bit_size > 0U) // constexpr
                result.parts[data_repr_parts_count - 1] &= last_part_mask;
            return result;
        }

        /// @brief Performs bitwise-AND of current object and another data_parts object. Updates current object.
        ///
        /// @param rhs Right-hand side.
        /// @return    Current instance of data_parts.
        data_parts& operator &=(const data_parts& rhs) noexcept
        {
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                parts[pi] &= rhs.parts[pi];
            return *this;
        }

        /// @brief Performs bitwise-OR of current object and another data_parts object. Updates current object.
        ///
        /// @param rhs Right-hand side.
        /// @return    Current instance of data_parts.
        data_parts& operator |=(const data_parts& rhs) noexcept
        {
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                parts[pi] |= rhs.parts[pi];
            return *this;
        }

        /// @brief Performs bitwise-XOR of current object and another data_parts object. Updates current object.
        ///
        /// @param rhs Right-hand side.
        /// @return    Current instance of data_parts.
        data_parts& operator ^=(const data_parts& rhs) noexcept
        {
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                parts[pi] ^= rhs.parts[pi];
            return *this;
        }

        /// @brief Performs bitwise-MINUS of current object and another data_parts object. Updates current object.
        /// @details Bitwise-MINUS is similar to difference of two sets, equivalent to:
        ///          <tt>A - B === A & ~B</tt>.
        ///
        /// @param rhs Right-hand side.
        /// @return    Current instance of data_parts.
        data_parts& operator -=(const data_parts& rhs) noexcept
        {
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
                parts[pi] &= ~rhs.parts[pi];
            if (last_part_bit_size > 0U) // constexpr
                parts[data_repr_parts_count - 1] &= last_part_mask;
            return *this;
        }

        // ------------------------------------------------------------------------------------------------------------
        // data_parts: EqualityComparable implementation.
        // ------------------------------------------------------------------------------------------------------------

        /// @brief Tests if mask parts are equal (same bits set).
        /// @param lhs Left side to compare.
        /// @param rhs Right side to compare.
        /// @return @c true if both sides are equal; otherwise, @c false.
        friend bool operator ==(const data_parts& lhs, const data_parts& rhs) noexcept
        {
            for (std::size_t pi = 0; pi < data_repr_parts_count; ++pi)
            {
                if (lhs.parts[pi] != rhs.parts[pi])
                    return false;
            }
            return true;
        }

        /// @brief Tests if mask parts are not equal (different bits set).
        /// @param lhs Left side to compare.
        /// @param rhs Right side to compare.
        /// @return @c true if sides are different; otherwise, @c false.
        friend bool operator !=(const data_parts& lhs, const data_parts& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        // ------------------------------------------------------------------------------------------------------------
        // data_parts: Data members.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Parts of a mask.
        part_type parts[data_repr_parts_count];
    };

public:
    /// @brief Marker type that allows to select correct candidate constructor (that creates mask by position
    ///        or position and length).
    struct create_by_pos_mark {};
    /// @brief Marker used to select correct overload constructor that specifies position (instead of possible
    ///        integral value of data type).
    static constexpr create_by_pos_mark create_by_pos{}; // "{}": WA for clang 3.6-3.9

private:
    // ----------------------------------------------------------------------------------------------------------------
    // Constructors, special functions, destructors.
    // ----------------------------------------------------------------------------------------------------------------
    /// @brief Creates mask from underlying data_parts object.
    ///
    /// @param mask Underlying storage with mask data.
    constexpr explicit data_type_mask(const data_parts& mask)
        : _mask(mask) {}

public:
    /// @brief Creates mask with all bits cleared (set to @c 0).
    constexpr data_type_mask() noexcept
        : _mask(data_parts_zero) {}

    /// @brief Creates mask initialized to binary representation of data type.
    ///
    /// @param val Value which binary representation will be used to initialize mask.
    // ReSharper disable once CppNonExplicitConvertingConstructor
    data_type_mask(const data_type& val) noexcept
    {
        union { data_type input_val; data_parts input_parts; }; // NOLINT(cppcoreguidelines-pro-type-member-init,
                                                                //        hicpp-member-init)

        input_val = val;
        _mask     = input_parts;

        if (last_part_bit_size > 0U) // constexpr
            _mask.parts[data_repr_parts_count - 1] &= last_part_mask;
    }

    /// @brief Creates mask with set bits starting from specified index (bits before index are cleared).
    ///
    /// @details Sets mask bits starting from specified index of bit. All bits starting from @p from are set
    /// (set to @c 1). Bits before @c from are cleared (set to @c 0).
    ///
    /// @param from Index (@c 0-based) indicating first bit to set.
    // ReSharper disable once CppDoxygenUndocumentedParameter
    data_type_mask(const std::size_t from, create_by_pos_mark) noexcept
        : _mask(data_parts_zero)
    {
        _mask.template set_bits<true>(from);
    }

    /// @brief Creates mask with set bits starting from specified index (bits before index are cleared).
    ///
    /// @details Sets mask bits starting from specified index of bit. All bits starting from @p from are set
    /// (set to @c 1). Bits before @c from are cleared (set to @c 0).
    ///
    /// @param from Index (@c 0-based) indicating first bit to set.
    /// @return     Mask with proper bits set (rest of bits is cleared).
    static data_type_mask create_from_position(const std::size_t from)
    {
        return data_type_mask(from, create_by_pos);
    }

    /// @brief Creates mask with set specifed length of bits starting from specified index
    /// (bits before and after set bits are cleared).
    ///
    /// @details Sets mask bits starting from specified index of bit. All bits starting from @p from to (exclusive)
    /// <tt>from + len</tt> are set (to @c 1). Bits before and after (including bit at position
    /// <tt>from + len</tt>) are cleared (set to @c 0).
    ///
    /// @param from Index (@c 0-based) indicating first bit to set.
    /// @param len  Number of bits to set.
    data_type_mask(const std::size_t from, const std::size_t len,
                                        // ReSharper disable once CppDoxygenUndocumentedParameter
                                        create_by_pos_mark) noexcept
        : _mask(data_parts_zero)
    {
        _mask.template set_bits<true>(from);
        _mask -= data_parts(data_parts_zero).template set_bits<true>(from + len);
    }

    /// @brief Creates mask with set specifed length of bits starting from specified index
    /// (bits before and after set bits are cleared).
    ///
    /// @details Sets mask bits starting from specified index of bit. All bits starting from @p from to (exclusive)
    /// <tt>from + len</tt> are set (to @c 1). Bits before and after (including bit at position
    /// <tt>from + len</tt>) are cleared (set to @c 0).
    ///
    /// @param from Index (@c 0-based) indicating first bit to set.
    /// @param len  Number of bits to set.
    /// @return     Mask with proper bits set (rest of bits is cleared).
    static data_type_mask create_from_position(const std::size_t from, const std::size_t len)
    {
        return data_type_mask(from, len, create_by_pos);
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Functions.
    // ----------------------------------------------------------------------------------------------------------------

    /// @brief Converts / reinterprets mask as data type.
    ///
    /// @return Data type with the same bit representation as mask.
    data_type to_data_type() const
    {
        union { data_type input_val; data_parts input_parts; }; // NOLINT(cppcoreguidelines-pro-type-member-init,
                                                                //        hicpp-member-init)

        input_parts = _mask;
        return input_val;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Operators.
    // ----------------------------------------------------------------------------------------------------------------
    /// @brief Negates all bits of the mask.
    ///
    /// @return Mask with all bits negated.
    data_type_mask operator ~() const noexcept
    {
        return data_type_mask(~_mask);
    }

    /// @brief Performs bitwise-AND of two masks.
    ///
    /// @param rhs Right-hand side.
    /// @return    Mask with resulting bitwise-AND bits.
    data_type_mask operator &(const data_type_mask& rhs) const noexcept
    {
        return data_type_mask(_mask & rhs._mask);
    }

    /// @brief Performs bitwise-OR of two masks.
    ///
    /// @param rhs Right-hand side.
    /// @return    Mask with resulting bitwise-OR bits.
    data_type_mask operator |(const data_type_mask& rhs) const noexcept
    {
        return data_type_mask(_mask | rhs._mask);
    }

    /// @brief Performs bitwise-XOR of two masks.
    ///
    /// @param rhs Right-hand side.
    /// @return    Mask with resulting bitwise-XOR bits.
    data_type_mask operator ^(const data_type_mask& rhs) const noexcept
    {
        return data_type_mask(_mask ^ rhs._mask);
    }

    /// @brief Performs bitwise-MINUS of two masks.
    /// @details Bitwise-MINUS is similar to difference of two sets, equivalent to:
    ///          <tt>A - B === A & ~B</tt>.
    ///
    /// @param rhs Right-hand side.
    /// @return    Mask with resulting bitwise-MINUS bits.
    data_type_mask operator -(const data_type_mask& rhs) const noexcept
    {
        return data_type_mask(_mask - rhs._mask);
    }

    /// @brief Performs bitwise-AND of current object and another mask object. Updates current object.
    ///
    /// @param rhs Right-hand side.
    /// @return    Current instance of mask.
    data_type_mask& operator &=(const data_type_mask& rhs) noexcept
    {
        _mask &= rhs._mask;
        return *this;
    }

    /// @brief Performs bitwise-OR of current object and another mask object. Updates current object.
    ///
    /// @param rhs Right-hand side.
    /// @return    Current instance of mask.
    data_type_mask& operator |=(const data_type_mask& rhs) noexcept
    {
        _mask |= rhs._mask;
        return *this;
    }

    /// @brief Performs bitwise-XOR of current object and another mask object. Updates current object.
    ///
    /// @param rhs Right-hand side.
    /// @return    Current instance of mask.
    data_type_mask& operator ^=(const data_type_mask& rhs) noexcept
    {
        _mask ^= rhs._mask;
        return *this;
    }

    /// @brief Performs bitwise-MINUS of current object and another mask object. Updates current object.
    /// @details Bitwise-MINUS is similar to difference of two sets, equivalent to:
    ///          <tt>A - B === A & ~B</tt>.
    ///
    /// @param rhs Right-hand side.
    /// @return    Current instance of mask.
    data_type_mask& operator -=(const data_type_mask& rhs) noexcept
    {
        _mask -= rhs._mask;
        return *this;
    }

    /// @brief Serialize to stream.
    ///
    /// @tparam CharType   Stream character type.
    /// @tparam CharTraits Stream character traits.
    /// @param os  Output stream.
    /// @param rhs Object to serialize.
    /// @return    Output stream. The same as @c os.
    template<typename CharType, typename CharTraits>
    friend std::basic_ostream<CharType, CharTraits>&
        operator <<(std::basic_ostream<CharType, CharTraits>& os, const data_type_mask& rhs)
    {
        auto guard = create_format_guard(os);

        os << std::setfill(os.widen('0')) << std::hex;
        if (little_endian)
        {
            if (last_part_bit_size > 0U)
                os << std::setw(last_part_bit_size / 4) << rhs._mask.part_hilo_ord(0);
            for (std::size_t pi = (last_part_bit_size > 0U); pi < data_repr_parts_count; ++pi)
                os << std::setw(data_part_bit_size / 4) << rhs._mask.part_hilo_ord(pi);
        }
        else
        {
            constexpr std::size_t full_parts_count = data_repr_parts_count - (last_part_bit_size > 0U);
            for (std::size_t pi = 0U; pi < full_parts_count; ++pi)
                os << std::setw(data_part_bit_size / 4) << rhs._mask.part_hilo_ord(pi);
            if (last_part_bit_size > 0U)
            {
                os << std::setw(last_part_bit_size / 4)
                    << (rhs._mask.part_hilo_ord(full_parts_count) >>
                            ((data_part_bit_size - last_part_bit_size) % data_part_bit_size));
            }
        }

        return os;
    }



    /// @brief Deserialize from stream.
    ///
    /// @tparam CharType   Stream character type.
    /// @tparam CharTraits Stream character traits.
    /// @param is  Input stream.
    /// @param rhs Placeholder for object to deserialize.
    /// @return    Input stream. The same as @c is.
    template<typename CharType, typename CharTraits>
    friend std::basic_istream<CharType, CharTraits>&
        operator >>(std::basic_istream<CharType, CharTraits>& is, data_type_mask& rhs)
    {
        using part_string_type = std::basic_string<CharType, CharTraits>;
        using part_stream_type = std::basic_istringstream<CharType, CharTraits>;

        auto guard = create_format_guard(is);

        data_parts mask;
        part_string_type part_string;
        part_stream_type part_stream;

        part_stream >> std::hex >> std::noskipws;
        if (little_endian)
        {
            if (last_part_bit_size > 0U)
            {
                is >> std::setw(last_part_bit_size / 4) >> part_string >> std::noskipws;
                part_stream.str(part_string);
                part_stream.seekg(0, std::ios::beg);
                part_stream >> mask.part_hilo_ord(0);

                if (!part_stream)
                    is.setstate(std::ios::failbit);
                if (!is) { return is; }
            }
            for (std::size_t pi = (last_part_bit_size > 0U); pi < data_repr_parts_count; ++pi)
            {
                is >> std::setw(data_part_bit_size / 4) >> part_string >> std::noskipws;
                part_stream.str(part_string);
                part_stream.seekg(0, std::ios::beg);
                part_stream >> mask.part_hilo_ord(pi);

                if (!part_stream)
                    is.setstate(std::ios::failbit);
                if (!is) { return is; }
            }
        }
        else
        {
            constexpr std::size_t full_parts_count = data_repr_parts_count - (last_part_bit_size > 0U);
            for (std::size_t pi = 0U; pi < full_parts_count; ++pi)
            {
                is >> std::setw(data_part_bit_size / 4) >> part_string >> std::noskipws;
                part_stream.str(part_string);
                part_stream.seekg(0, std::ios::beg);
                part_stream >> mask.part_hilo_ord(pi);

                if (!part_stream)
                    is.setstate(std::ios::failbit);
                if (!is) { return is; }
            }
            if (last_part_bit_size > 0U)
            {
                is >> std::setw(last_part_bit_size / 4) >> part_string >> std::noskipws;
                part_stream.str(part_string);
                part_stream.seekg(0, std::ios::beg);
                part_stream >> mask.part_hilo_ord(full_parts_count);
                mask.part_hilo_ord(full_parts_count) <<= ((data_part_bit_size - last_part_bit_size) % data_part_bit_size);

                if (!part_stream)
                    is.setstate(std::ios::failbit);
                if (!is) { return is; }
            }
        }
        if (last_part_bit_size > 0U) // constexpr
            mask.parts[data_repr_parts_count - 1] &= last_part_mask;

        rhs._mask = mask;
        return is;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // EqualityComparable implementation.
    // ----------------------------------------------------------------------------------------------------------------

    /// @brief Tests if masks are equal (same bits set).
    /// @param lhs Left side to compare.
    /// @param rhs Right side to compare.
    /// @return @c true if both sides are equal; otherwise, @c false.
    friend bool operator ==(const data_type_mask& lhs, const data_type_mask& rhs) noexcept
    {
        return lhs._mask == rhs._mask;
    }

    /// @brief Tests if masks are not equal (different bits set).
    /// @param lhs Left side to compare.
    /// @param rhs Right side to compare.
    /// @return @c true if sides are different; otherwise, @c false.
    friend bool operator !=(const data_type_mask& lhs, const data_type_mask& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Data members.
    // ----------------------------------------------------------------------------------------------------------------
private:
    data_parts _mask; ///< Mask bits.
};

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

} // namespace detail
/// @endcond

/// @brief Uniform real distribution that generates values in range <tt>[a; b]</tt>.
///
/// @details Uniform distribution with following properties:
///          @li Generated values are in range <tt>[a; b]</tt> (inclusive from both sides).
///          @li Values are mapped from uniform distribution <tt>[1; 2]</tt> where significand part
///              of generated number is randomized only on specified number of highest bits
///              (specified by @p significand_rand_bits). The rest of bits in signficand is set
///              to @c 0.
///
///          The type meets @a CopyConstructible, @a CopyAssignable, @a MoveConstructable, @a MoveAssignable,
///          @a EqualityComparable and @a DefaultConstructible requirements.
///
///          The type meets all requirements of C++11/14 random number distribution.
///
/// @tparam RealType Type of real numbers generated by distribution. The type must be floating point type
///                  compatible with IEC 559 (IEEE 754/854) standard and use radix of @c 2.
template <typename RealType = float>
class uniform_quantized_real_distribution
{
    // ----------------------------------------------------------------------------------------------------------------
    // Compile-time tests.
    // ----------------------------------------------------------------------------------------------------------------
    static_assert(std::is_floating_point<RealType>::value,
                  "RealType must be floating-point type.");
    static_assert(std::numeric_limits<RealType>::is_iec559,
                  "RealType must have representation compatible with IEC 559 / IEEE 754 / IEEE 854");
    static_assert(std::numeric_limits<RealType>::radix == 2,
                  "Only binary-based floating-point types are supported as RealType.");

    // ----------------------------------------------------------------------------------------------------------------
    // Typedefs, aliases and constants (including nested types and their helpers).
    // ----------------------------------------------------------------------------------------------------------------
public:
    /// @brief Type of generated elements by distribution.
    using result_type = RealType;
private:
    /// @brief Type of mask used to filter significant bits from result representation.
    using mask_type            = detail::data_type_mask<result_type>;
    /// @brief Type of underlying distribution used to generate results.
    using underlying_dist_type = std::uniform_real_distribution<result_type>;

public:
    /// @brief Value of result_type that is equivalent to @c 0.
    static constexpr result_type result_zero = static_cast<result_type>(0);
    /// @brief Value of result_type that is equivalent to @c 1.
    static constexpr result_type result_one = static_cast<result_type>(1);
private:
    /// @brief Value of result_type that is equivalent to @c 2.
    static constexpr result_type result_two = static_cast<result_type>(2);
    /// @brief Maximum finite value of result_type.
    static constexpr result_type result_max = std::numeric_limits<result_type>::max();

public:
    /// @brief Number of bits in significand part of current floating-point type.
    ///
    /// @details Specified number of bits counts only bits existing in binary representation
    ///          (hidden implicit bits are not included).
    static constexpr unsigned significand_max_bits = ((std::numeric_limits<result_type>::digits - 1) > 0) ?
                                                       std::numeric_limits<result_type>::digits - 1 : 0;

    /// @brief Type of parameters used by distribution.
    ///
    /// @details The type meets @a CopyConstructible, @a CopyAssignable, @a MoveConstructable, @a MoveAssignable,
    ///          @a EqualityComparable and @a DefaultConstructible requirements.
    struct param_type
    {
        // ------------------------------------------------------------------------------------------------------------
        // param_type: Typedefs, aliases and constants (including nested types).
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Type of distribution for which current parameters are used.
        using distribution_type = uniform_quantized_real_distribution;

        // ------------------------------------------------------------------------------------------------------------
        // param_type: Constructors, special functions, destructors.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Creates uniform distribution parameters that generate values in range <tt>[a; b]</tt>.
        ///
        /// @details Creates parameters that describe uniform distribution with following properties:
        ///          @li Generated values are in range <tt>[a; b]</tt> (inclusive from both sides).
        ///          @li Values are mapped from uniform distribution <tt>[1; 2]</tt> where significand part
        ///              of generated number is randomized only on specified number of highest bits
        ///              (specified by @p significand_rand_bits). The rest of bits in signficand is set
        ///              to @c 0.
        ///
        /// @param a                     Inclusive lower bound for generated values. Each generated value is
        ///                              greater than or equal to @p a.
        /// @param b                     Inclusive upper bound for generated values. Each generated value is
        ///                              less than or equal to @p a.
        /// @param significand_rand_bits Number of high bits to randomize in significand part of floating-point
        ///                              value. The value is clamped from above by significand_max_bits.
        ///
        /// @throw std::invalid_argument Lower bound (@p a) is not finite number (infinite or @c NaN).
        /// @throw std::invalid_argument Upper bound (@p b) is not finite number (infinite or @c NaN).
        /// @throw std::invalid_argument Lower bound (@p a) is greater than upper bound (@p b).
        explicit param_type(
            const result_type a                  = result_zero,
            const result_type b                  = result_one,
            const unsigned significand_rand_bits = significand_max_bits)
            // NOTE: The "+" before significand_max_bits is important since the value is odr-used and we
            //       need actual storage (address of it). With unary + operator we can create temporary.
            : _a(a), _b(b), _significand_rand_bits(std::min(significand_rand_bits, +significand_max_bits))
        {
            if (!std::isfinite(a))
                throw std::invalid_argument("Lower bound (a) must be finite.");
            if (!std::isfinite(b))
                throw std::invalid_argument("Upper bound (b) must be finite.");
            if (a > b)
                throw std::invalid_argument("Lower bound (a) must be less than or equal to upper bound (b).");
        }

        /// @brief Creates uniform distribution parameters that generate values in range <tt>[0; 1]</tt>.
        ///
        /// @details Creates parameters that desribe uniform distribution with following properties:
        ///          @li Generated values are in range <tt>[0; 1]</tt> (inclusive from both sides).
        ///          @li Values are mapped from uniform distribution <tt>[1; 2]</tt> where significand part
        ///              of generated number is randomized only on specified number of highest bits
        ///              (specified by @p significand_rand_bits). The rest of bits in signficand is set
        ///              to @c 0.
        ///
        /// @param significand_rand_bits Number of high bits to randomize in significand part of floating-point
        ///                              value. The value is clamped from above by significand_max_bits.
        // ReSharper disable once CppPossiblyUninitializedMember
        explicit param_type(const unsigned significand_rand_bits)
            : param_type(result_zero, result_one, significand_rand_bits)
        {}

        // ------------------------------------------------------------------------------------------------------------
        // param_type: Properties, Accessors.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Gets lower bound.
        ///
        /// @return Inclusive lower bound for generated numbers. It is finite and lower-equal to b().
        result_type a() const
        {
            return _a;
        }

        /// @brief Gets upper bound.
        ///
        /// @return Inclusive upper bound for generated numbers. It is finite and greater-equal to a().
        result_type b() const
        {
            return _b;
        }

        /// @brief Gets number of bits randomized in generated floating-point numbers.
        ///
        /// @return Number of high bits randomized in significand part of floating-point value.
        unsigned significand_rand_bits() const
        {
            return _significand_rand_bits;
        }

        // ------------------------------------------------------------------------------------------------------------
        // param_type: EqualityComparable implementation.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Tests if distribution parameter sets are equal (same content).
        /// @param lhs Left side to compare.
        /// @param rhs Right side to compare.
        /// @return @c true if both sides are equal; otherwise, @c false.
        friend bool operator ==(const param_type& lhs, const param_type& rhs)
        {
            return lhs._a == rhs._a
                && lhs._b == rhs._b
                && lhs._significand_rand_bits == rhs._significand_rand_bits;
        }

        /// @brief Tests if distribution parameter sets are not equal (different content).
        /// @param lhs Left side to compare.
        /// @param rhs Right side to compare.
        /// @return @c true if sides are different; otherwise, @c false.
        friend bool operator !=(const param_type& lhs, const param_type& rhs)
        {
            return !(lhs == rhs);
        }

        // ------------------------------------------------------------------------------------------------------------
        // param_type: Data members.
        // ------------------------------------------------------------------------------------------------------------
    private:
        result_type _a;                  ///< Inclusive lower bound for generated values.
        result_type _b;                  ///< Inclusive upper bound for generated values.
        unsigned _significand_rand_bits; ///< Number of high bits to randomize in signficand part.
    };

private:
    /// @brief RAII guard that allows to temporary swap distribution parameters.
    ///
    /// @details RAII guard allows to swap parameters of distribution. It restores
    ///          original ones when it is destroyed.
    ///
    /// @tparam Distribution Distribution type which meets @a RandomNumberDistribution
    ///                      trait.
    template <typename Distribution>
    struct param_swap_guard
    {
        // ------------------------------------------------------------------------------------------------------------
        // param_swap_guard: Typedefs, aliases and constants (including nested types).
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Type of distribution handled by current guard.
        using distribution_type = Distribution;
        /// @brief Type of parameter set swapped and restored by current guard.
        using param_type        = typename distribution_type::param_type;

        // ------------------------------------------------------------------------------------------------------------
        // param_swap_guard: Constructors, special functions, destructors.
        // ------------------------------------------------------------------------------------------------------------
        /// @brief Creates guard and swaps parameters of specified distribution to new parameter set.
        ///
        /// @details Guard stores original parameters before swapping and restores them upon destruction.
        ///
        /// @param distribution Distribution instance which parameters will be swapped temporarily.
        /// @param new_param    New parameter set for distribution.
        param_swap_guard(distribution_type& distribution, const param_type& new_param)
            : _distibution(distribution), _old_param(distribution.param())
        {
            _distibution.param(new_param);
        }

        // Explicitly deleted/defaulted special functions:
        param_swap_guard(const param_swap_guard& other) = delete;
        param_swap_guard(param_swap_guard&& other) noexcept = default;
        param_swap_guard& operator =(const param_swap_guard& other) = delete;
        param_swap_guard& operator =(param_swap_guard&& other) noexcept = delete;

        /// @brief Destroys guard and restores parameters of distribution handled by current guard.
        ~param_swap_guard()
        {
            _distibution.param(_old_param);
        }

        // ------------------------------------------------------------------------------------------------------------
        // param_swap_guard: Data members.
        // ------------------------------------------------------------------------------------------------------------
    private:
        distribution_type& _distibution; ///< Distribution instance which parameters are temporary swapped.
        param_type         _old_param;   ///< Original value of parameter set (before swap).
    };

    /// @brief Temporarily swaps parameters in distribution and returns guard object which
    ///        will restore original parameters upon destruction.
    ///
    /// @tparam Distribution Distribution type which meets @a RandomNumberDistribution
    ///                      trait.
    /// @param distribution Distribution instance which parameters will be swapped temporarily.
    /// @param new_param    New parameter set for distribution.
    /// @return             Guard object that will restore distribution to original state
    ///                     at the end of life (during destruction).
    template <typename Distribution>
    param_swap_guard<Distribution> temporary_swap_param(
        Distribution& distribution,
        const typename Distribution::param_type& new_param)
    {
        return param_swap_guard<Distribution>(distribution, new_param);
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Helper functions.
    // ----------------------------------------------------------------------------------------------------------------
    /// @brief Creates internal mask for representation (to filter significand bits).
    ///
    /// @param param Distribution parameter set.
    /// @return Internal mask.
    static mask_type create_mask(const param_type& param)
    {
        return mask_type(significand_max_bits - param.significand_rand_bits(), mask_type::create_by_pos);
    }

    /// @brief Creates underlying distribution parameter set.
    ///
    /// @details Ensures that upper bound has the same probability to occur as rest of values (when clamped by
    ///          @c 2).
    ///
    /// @param mask Internal mask used.
    /// @return     Parameter set of underlying distribution.
    static typename underlying_dist_type::param_type create_underlying_dist_param(const mask_type& mask)
    {
        // NOTE: The "+" before result_one is important since the value is odr-used and we
        //       need actual storage (address of it). With unary + operator we can create temporary.
        const result_type upper_bound =
            std::nextafter((mask_type(+result_one) | ~mask).to_data_type(), result_max) + result_one;

        return underlying_dist_type(result_one, upper_bound).param();
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Constructors, special functions, destructors.
    // ----------------------------------------------------------------------------------------------------------------
public:
    /// @brief Creates uniform distribution that generates values in range <tt>[a; b]</tt>.
    ///
    /// @details Creates uniform distribution with following properties:
    ///          @li Generated values are in range <tt>[a; b]</tt> (inclusive from both sides).
    ///          @li Values are mapped from uniform distribution <tt>[1; 2]</tt> where significand part
    ///              of generated number is randomized only on specified number of highest bits
    ///              (specified by @p significand_rand_bits). The rest of bits in signficand is set
    ///              to @c 0.
    ///
    /// @param a                     Inclusive lower bound for generated values. Each generated value is
    ///                              greater than or equal to @p a.
    /// @param b                     Inclusive upper bound for generated values. Each generated value is
    ///                              less than or equal to @p a.
    /// @param significand_rand_bits Number of high bits to randomize in significand part of floating-point
    ///                              value. The value is clamped from above by significand_max_bits.
    ///
    /// @throw std::invalid_argument Lower bound (@p a) is not finite number (infinite or @c NaN).
    /// @throw std::invalid_argument Upper bound (@p b) is not finite number (infinite or @c NaN).
    /// @throw std::invalid_argument Lower bound (@p a) is greater than upper bound (@p b).
    explicit uniform_quantized_real_distribution(
        const result_type a                  = result_zero,
        const result_type b                  = result_one,
        const unsigned significand_rand_bits = significand_max_bits)
        : uniform_quantized_real_distribution(param_type(a, b, significand_rand_bits))
    {
    }

    /// @brief Creates uniform distribution that generates values in range <tt>[0; 1]</tt>.
    ///
    /// @details Creates uniform distribution with following properties:
    ///          @li Generated values are in range <tt>[0; 1]</tt> (inclusive from both sides).
    ///          @li Values are mapped from uniform distribution <tt>[1; 2]</tt> where significand part
    ///              of generated number is randomized only on specified number of highest bits
    ///              (specified by @p significand_rand_bits). The rest of bits in signficand is set
    ///              to @c 0.
    ///
    /// @param significand_rand_bits Number of high bits to randomize in significand part of floating-point
    ///                              value. The value is clamped from above by significand_max_bits.
    explicit uniform_quantized_real_distribution(
        const unsigned significand_rand_bits)
        : uniform_quantized_real_distribution(param_type(significand_rand_bits))
    {}

    /// @brief Creates uniform distribution based on distribution parameter set.
    ///
    /// @details Creates uniform distribution with following properties:
    ///          @li Generated values are in range <tt>[param_type::a(); param_type::b()]</tt>
    ///              (inclusive from both sides).
    ///          @li Values are mapped from uniform distribution <tt>[1; 2]</tt> where significand part
    ///              of generated number is randomized only on specified number of highest bits
    ///              (specified by param_type::significand_rand_bits()). The rest of bits
    ///              in signficand is set to @c 0.
    ///
    /// @param param Parameter set containing all parameters required to create distribution.
    // ReSharper disable once CppNonExplicitConvertingConstructor
    uniform_quantized_real_distribution(
        const param_type& param)
        : _param(param),
          _mask(create_mask(_param)),
          _base_distribution(create_underlying_dist_param(_mask))
    {}

    // ----------------------------------------------------------------------------------------------------------------
    // State and generators.
    // ----------------------------------------------------------------------------------------------------------------
    /// @brief Resets internal state of distribution, so it will return return numbers
    ///        independent of previous states of generator.
    void reset()
    {
        _base_distribution.reset();
    }

    /// @brief Generates random value.
    ///
    /// @tparam Generator Random number generator type used to generate random bits.
    /// @param generator  Random number generator instance used to generate random bits.
    /// @return           Random value. See description of distribution to get detailed
    ///                   information about numbers generated.
    template <typename Generator>
    result_type operator ()(Generator& generator)
    {
        if (_param.a() == _param.b())
            return _param.a();

        result_type rnd_val        = _base_distribution(generator);
        result_type mask_rnd_val   = (_mask & rnd_val).to_data_type();
        result_type scaled_rnd_val = _param.a() + (mask_rnd_val - result_one) * (_param.b() - _param.a());

        return scaled_rnd_val;
    }

    /// @brief Generates random value.
    ///
    /// @tparam Generator Random number generator type used to generate random bits.
    /// @param generator  Random number generator instance used to generate random bits.
    /// @param param      Different parameter set used to change characteristics of generated numbers.
    /// @return           Random value. See description of distribution to get detailed
    ///                   information about numbers generated.
    template <typename Generator>
    result_type operator ()(Generator& generator, const param_type& param)
    {
        auto guard = temporary_swap_param(*this, param);

        return (*this)(generator);
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Properties, Accessors.
    // ----------------------------------------------------------------------------------------------------------------
    /// @brief Gets parameters of distribution (as a single parameter set).
    ///
    /// @return Parameter set containing all parameters required to describe distribution.
    param_type param() const
    {
        return _param;
    }

    /// @brief Sets parameters of distribution to a new value.
    ///
    /// @param param Parameter set containing all parameters required to describe distribution.
    void param(const param_type& param)
    {
        _param = param;
        _mask = create_mask(_param);
        _base_distribution.param(create_underlying_dist_param(_mask));
    }

    /// @brief Gets minimum value that can be returned by distribution.
    ///
    /// @return Distribution lower bound (inclusive).
    result_type min() const
    {
        return a();
    }

    /// @brief Gets maximum value that can be returned by distribution.
    ///
    /// @return Distribution upper bound (inclusive).
    result_type max() const
    {
        return b();
    }

    // ----------------------------------------------------------------------------------------------------------------

    /// @brief Gets lower bound.
    ///
    /// @return Inclusive lower bound for generated numbers. It is finite and lower-equal to b().
    result_type a() const
    {
        return _param.a();
    }

    /// @brief Gets upper bound.
    ///
    /// @return Inclusive upper bound for generated numbers. It is finite and greater-equal to a().
    result_type b() const
    {
        return _param.b();
    }

    /// @brief Gets number of bits randomized in generated floating-point numbers.
    ///
    /// @return Number of high bits randomized in significand part of floating-point value.
    unsigned significand_rand_bits() const
    {
        return _param.significand_rand_bits();
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Operators.
    // ----------------------------------------------------------------------------------------------------------------

    /// @brief Serialize to stream.
    ///
    /// @tparam CharType   Stream character type.
    /// @tparam CharTraits Stream character traits.
    /// @param os  Output stream.
    /// @param rhs Object to serialize.
    /// @return    Output stream. The same as @c os.
    template<typename CharType, typename CharTraits>
    friend std::basic_ostream<CharType, CharTraits>&
        operator <<(std::basic_ostream<CharType, CharTraits>& os,
                    const uniform_quantized_real_distribution& rhs)
    {
        using stream_type = std::basic_ostringstream<CharType, CharTraits>;

        // Although the serialized members are named, the order must be preserved.
        const char hdr_a[]   = "{ A: ";
        const char hdr_b[]   = ", B: ";
        const char hdr_srb[] = ", SRB: ";
        const char hdr_m[]   = ", M: ";
        const char hdr_d[]   = ", BD: { ";
        const char ftr[]     = " } }";

        auto guard = detail::create_format_guard(os);

        stream_type ss;
        ss << hdr_a << mask_type(rhs._param.a())
            << hdr_b << mask_type(rhs._param.b())
            << hdr_srb << rhs._param.significand_rand_bits()
            << hdr_m << rhs._mask
            << hdr_d << rhs._base_distribution
            << ftr;

        return os << ss.str();
    }

private:
    /// @brief Ignores characters until specified character is encountered (or EOF is reached).
    ///
    /// @details The specified character is also discarded.
    ///
    /// @tparam CharType   Stream character type.
    /// @tparam CharTraits Stream character traits.
    /// @param is Input stream.
    /// @param c  Character until which all characters should be ignored. The specified character is also discarded.
    /// @return   Input stream. The same as @c is.
    template<typename CharType, typename CharTraits>
    static std::basic_istream<CharType, CharTraits>& skip_to_char(std::basic_istream<CharType, CharTraits>& is, char c)
    {
        constexpr std::streamsize ignore_max = std::numeric_limits<std::streamsize>::max();

        return is.ignore(ignore_max, CharTraits::to_int_type(is.widen(c)));
    }

public:
    /// @brief Deserialize from stream.
    ///
    /// @tparam CharType   Stream character type.
    /// @tparam CharTraits Stream character traits.
    /// @param is  Input stream.
    /// @param rhs Placeholder for object to deserialize.
    /// @return    Input stream. The same as @c is.
    template<typename CharType, typename CharTraits>
    friend std::basic_istream<CharType, CharTraits>&
        operator >>(std::basic_istream<CharType, CharTraits>& is,
                    uniform_quantized_real_distribution& rhs)
    {
        // Although the serialized members are named, the order must be preserved.
        auto guard = detail::create_format_guard(is);

        mask_type param_helper_a, param_helper_b, param_helper_m;
        unsigned param_helper_srb;
        underlying_dist_type param_helper_bd;

        rhs.skip_to_char(is, '{');
        rhs.skip_to_char(is, ':'); // A
        is >> std::ws >> param_helper_a;

        rhs.skip_to_char(is, ',');
        rhs.skip_to_char(is, ':'); // B
        is >> std::ws >> param_helper_b;

        rhs.skip_to_char(is, ',');
        rhs.skip_to_char(is, ':'); // SRB
        is >> std::ws >> param_helper_srb;

        param_type param(param_helper_a.to_data_type(), param_helper_b.to_data_type(), param_helper_srb);
        mask_type mask = create_mask(param);

        rhs.skip_to_char(is, ',');
        rhs.skip_to_char(is, ':'); // M
        is >> std::ws >> param_helper_m;

        rhs.skip_to_char(is, ',');
        rhs.skip_to_char(is, ':'); // BD
        rhs.skip_to_char(is, '{');
        is >> std::ws >> param_helper_bd;

        rhs.skip_to_char(is, '}');
        rhs.skip_to_char(is, '}');

        if (mask != param_helper_m)
            is.setstate(std::ios::failbit);
        if (!is) { return is; }

        rhs._param             = param;
        rhs._mask              = mask;
        rhs._base_distribution = param_helper_bd;
        return is;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // EqualityComparable implementation.
    // ----------------------------------------------------------------------------------------------------------------
    /// @brief Tests if distributions are equal (same configuration).
    /// @param lhs Left side to compare.
    /// @param rhs Right side to compare.
    /// @return @c true if both sides are equal; otherwise, @c false.
    friend bool operator ==(const uniform_quantized_real_distribution& lhs,
                            const uniform_quantized_real_distribution& rhs)
    {
        return lhs._param == rhs._param
            && lhs._mask == rhs._mask
            && lhs._base_distribution == rhs._base_distribution;
    }

    /// @brief Tests if distributions are not equal (different configuration).
    /// @param lhs Left side to compare.
    /// @param rhs Right side to compare.
    /// @return @c true if sides are different; otherwise, @c false.
    friend bool operator !=(const uniform_quantized_real_distribution& lhs,
                            const uniform_quantized_real_distribution& rhs)
    {
        return !(lhs == rhs);
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Data members.
    // ----------------------------------------------------------------------------------------------------------------
private:
    param_type _param;                       ///< Parameters of distribution.
    mask_type _mask;                         ///< Mask that filters out unneeded bits from values generated by
                                             ///< underlying distribution.
    underlying_dist_type _base_distribution; ///< Base distribution used to generate input numbers.
};

}  // namespace distributions
}  // namespace tests
