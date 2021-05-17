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
                inline void set_bit(uint8_t* buf, size_t idx)
                {
                    const int byte_idx = idx / 8;
                    const int bit_idx = 7 - (idx % 8);
                    buf[byte_idx] |= (1 << bit_idx);
                }

                inline uint8_t get_bit(const uint8_t* buf, size_t idx)
                {
                    const int byte_idx = idx / 8;
                    const int bit_idx = 7 - (idx % 8);
                    return buf[byte_idx] & (1 << bit_idx);
                }
            } // namespace detail

            template <typename TI, typename TO>
            void convert(const TI* arg,
                         TO* out,
                         size_t count,
                         element::Type_t src_type,
                         element::Type_t dst_type)
            {
                std::memset(out, 0, count * sizeof(TO));

                if (dst_type == element::u1)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        if (arg[i])
                        {
                            detail::set_bit(reinterpret_cast<uint8_t*>(out), i);
                        }
                    }
                }
                else if (src_type == element::u1)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        if (detail::get_bit(reinterpret_cast<const uint8_t*>(arg), i))
                        {
                            out[i] = static_cast<TO>(1);
                        }
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
