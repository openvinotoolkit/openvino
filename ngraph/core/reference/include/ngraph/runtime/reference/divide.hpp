// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/runtime/reference/autobroadcast_binop.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // NOTE: Execution throws `std::domain_error` if either a non-integral value or an
            // out-of-bounds value is detected in the input tensor.

            // In English: return type is void and T must be an integral type.
            template <typename T>
            typename std::enable_if<std::is_integral<T>::value>::type
                divide(const T* arg0, const T* arg1, T* out, size_t count, bool pythondiv)
            {
                if (pythondiv)
                {
                    for (size_t i = 0; i < count; i++)
                    {
                        if (arg1[i] == 0)
                        {
                            throw std::domain_error("integer division by zero");
                        }
                        T quot = arg0[i] / arg1[i];
                        T rem = arg0[i] % arg1[i];
                        if ((rem != 0) && ((arg0[i] < 0) != (arg1[i] < 0)))
                        {
                            out[i] = quot - 1;
                        }
                        else
                        {
                            out[i] = quot;
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; i++)
                    {
                        if (arg1[i] == 0)
                        {
                            throw std::domain_error("integer division by zero");
                        }
                        out[i] = arg0[i] / arg1[i];
                    }
                }
            }

            template <typename T>
            typename std::enable_if<std::is_integral<T>::value>::type
                divide(const T* arg0,
                       const T* arg1,
                       T* out,
                       const Shape& arg0_shape,
                       const Shape& arg1_shape,
                       const op::AutoBroadcastSpec& broadcast_spec,
                       bool pythondiv)
            {
                auto functor = [pythondiv](T x, T y) -> T {
                    if (pythondiv)
                    {
                        if (y == 0)
                        {
                            throw std::domain_error("integer division by zero");
                        }
                        T quot = x / y;
                        T rem = x % y;
                        if ((rem != 0) && ((x < 0) != (y < 0)))
                        {
                            return quot - 1;
                        }
                        else
                        {
                            return quot;
                        }
                    }
                    else
                    {
                        if (y == 0)
                        {
                            throw std::domain_error("integer division by zero");
                        }
                        return x / y;
                    }
                };
                autobroadcast_binop(
                    arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, functor);
            }

            // In English: return type is void and T must be a standard floating point type, or
            // bfloat16, or float16.
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value ||
                                    std::is_same<T, bfloat16>::value ||
                                    std::is_same<T, float16>::value>::type
                divide(const T* arg0, const T* arg1, T* out, size_t count, bool pythondiv)
            {
                (void)pythondiv;
                for (size_t i = 0; i < count; i++)
                {
                    // TODO: Here we do not check for div by zero, so we'll get +-inf here
                    // if arg1[i] == 0. Is that the right thing to do? Jury's still out.
                    out[i] = arg0[i] / arg1[i];
                }
            }

            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value ||
                                    std::is_same<T, bfloat16>::value ||
                                    std::is_same<T, float16>::value>::type
                divide(const T* arg0,
                       const T* arg1,
                       T* out,
                       const Shape& arg0_shape,
                       const Shape& arg1_shape,
                       const op::AutoBroadcastSpec& broadcast_spec,
                       bool pythondiv)
            {
                (void)pythondiv;
                autobroadcast_binop(
                    arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> T {
                        return x / y;
                    });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
