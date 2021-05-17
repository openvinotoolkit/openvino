// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "ngraph/type/element_type.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace detail
            {
                inline void set_u1(uint8_t* buf, size_t idx)
                {
                    const int byte_idx = idx / 8;
                    const int bit_idx = 7 - (idx % 8);
                    buf[byte_idx] |= (1 << bit_idx);
                }

                inline uint8_t get_u1(const uint8_t* buf, size_t idx)
                {
                    const int byte_idx = idx / 8;
                    const int bit_idx = 7 - (idx % 8);
                    return buf[byte_idx] & (1 << bit_idx);
                }

                inline void set_u4(uint8_t* buf, size_t idx, uint8_t val)
                {
                    const int byte_idx = idx / 2;
                    const int bit_shift = 4 * (++idx % 2);
                    buf[byte_idx] &= ~(0xF << bit_shift); // half byte zeroed
                    buf[byte_idx] |= (val << bit_shift);  // set 1's
                }

                inline uint8_t get_u4(const uint8_t* buf, size_t idx)
                {
                    const int byte_idx = idx / 2;
                    const int bit_shift = 4 * (++idx % 2);
                    return (buf[byte_idx] >> bit_shift) & 0xF;
                }

                inline void set_i4(uint8_t* buf, size_t idx, int8_t val)
                {
                    const int byte_idx = idx / 2;
                    const int bit_shift = 4 * (++idx % 2);
                    buf[byte_idx] &= ~(0xF << bit_shift); // half byte zeroed
                    buf[byte_idx] |= (val << bit_shift);  // set 1's
                }

                inline int8_t get_i4(const uint8_t* buf, size_t idx)
                {
                    const int byte_idx = idx / 2;
                    const int bit_shift = 4 * (++idx % 2);
                    uint8_t val = (buf[byte_idx] >> bit_shift) & 0xF;
                    if (val & 0x08)
                    { // negative number
                        val |= 0xF0;
                    }
                    return val;
                }
            } // namespace detail

            template <typename TI, typename TO>
            void convert(const TI* arg,
                         TO* out,
                         size_t count,
                         element::Type_t src_type,
                         element::Type_t dst_type)
            {
                std::fill(out, out + count, 0);
                if (dst_type == element::u1) // TODO: fix for LP source types
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        if (arg[i])
                        {
                            detail::set_u1(reinterpret_cast<uint8_t*>(out), i);
                        }
                    }
                }
                else if (src_type == element::u1) // TODO: fix for LP dst types
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        if (detail::get_u1(reinterpret_cast<const uint8_t*>(arg), i))
                        {
                            out[i] = static_cast<TO>(1);
                        }
                    }
                }
                else if (dst_type == element::u4) // TODO: fix for LP source types
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        detail::set_u4(reinterpret_cast<uint8_t*>(out), i, arg[i]);
                    }
                }
                else if (src_type == element::u4) // TODO: fix for LP dst types
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        out[i] = detail::get_u4(reinterpret_cast<const uint8_t*>(arg), i);
                    }
                }
                else if (dst_type == element::i4) // TODO: fix for LP source types
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        detail::set_i4(reinterpret_cast<uint8_t*>(out), i, arg[i]);
                    }
                }
                else if (src_type == element::i4) // TODO: fix for LP dst types
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        out[i] = detail::get_i4(reinterpret_cast<const uint8_t*>(arg), i);
                    }
                }
                else
                {
                    NGRAPH_CHECK(false, "Unimplemented");
                }
            }

            template <typename TI, typename TO>
            typename std::enable_if<!std::is_same<TO, char>::value>::type
                convert(const TI* arg, TO* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = static_cast<TO>(arg[i]);
                }
            }

            template <>
            void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count);
            template <>
            void convert<float16, float>(const float16* arg, float* out, size_t count);
            template <>
            void convert<float, int8_t>(const float* arg, int8_t* out, size_t count);
            template <>
            void convert<float16, int8_t>(const float16* arg, int8_t* out, size_t count);

            // overload to handle ngraph::boolean (it is stored as char)
            template <typename TI, typename TO>
            typename std::enable_if<std::is_same<TO, char>::value>::type
                convert(const TI* arg, TO* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = static_cast<char>(static_cast<bool>(arg[i]));
                }
            }
        } // namespace reference

    } // namespace runtime

} // namespace ngraph
