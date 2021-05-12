// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            struct widen
            {
                using type = T;
            };

            template <>
            struct widen<float>
            {
                using type = double;
            };

            template <>
            struct widen<double>
            {
                using type = long double;
            };
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
