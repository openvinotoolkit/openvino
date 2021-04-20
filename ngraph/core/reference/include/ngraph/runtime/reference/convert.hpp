// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
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
