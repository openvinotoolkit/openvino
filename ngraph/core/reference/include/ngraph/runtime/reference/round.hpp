// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            T round_to_nearest_even(const T arg)
            {
                const auto floor_arg = std::floor(arg);
                const auto diff = arg - floor_arg;
                if (diff < 0.5f || (diff == 0.5f && static_cast<int>(floor_arg) % 2 == 0))
                {
                    return floor_arg;
                }
                else
                {
                    return floor_arg + 1.0f;
                }
            }

            template <typename T>
            void round(const T* arg, T* out, size_t count, const op::v5::Round::RoundMode mode)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    if (mode == op::v5::Round::RoundMode::HALF_TO_EVEN)
                    {
                        out[i] = round_to_nearest_even(arg[i]);
                    }
                    else
                    {
                        out[i] = std::round(arg[i]);
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
