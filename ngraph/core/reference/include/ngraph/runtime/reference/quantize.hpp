// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename REAL, typename QUANT>
            void quantize(const REAL* input,
                          const REAL* scale,
                          const QUANT* zero_point,
                          QUANT* output,
                          const Shape& input_shape,
                          const Shape& scale_zero_point_shape,
                          const AxisSet& axes,
                          op::Quantize::RoundMode round_mode)
            {
                CoordinateTransform input_transform(input_shape);
                CoordinateTransform scale_zero_point_transform(scale_zero_point_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate scale_zero_point_coord = project(input_coord, axes);

                    // apply scale
                    REAL qvalue = input[input_transform.index(input_coord)] /
                                  scale[scale_zero_point_transform.index(scale_zero_point_coord)];

                    // round
                    if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY)
                    {
                        REAL abs_qvalue = std::fabs(qvalue);
                        REAL abs_qvalue_toward_inf =
                            std::floor(abs_qvalue + static_cast<REAL>(0.5));
                        qvalue = (qvalue < static_cast<REAL>(0.0)) ? -abs_qvalue_toward_inf
                                                                   : abs_qvalue_toward_inf;
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO)
                    {
                        auto abs_qvalue = std::fabs(qvalue);
                        auto abs_qvalue_toward_zero =
                            std::ceil(abs_qvalue - static_cast<REAL>(0.5));
                        qvalue = (qvalue < static_cast<REAL>(0.0)) ? -abs_qvalue_toward_zero
                                                                   : abs_qvalue_toward_zero;
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_UPWARD)
                    {
                        qvalue = std::floor(qvalue + static_cast<REAL>(0.5));
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD)
                    {
                        qvalue = std::ceil(qvalue - static_cast<REAL>(0.5));
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)
                    {
                        auto up_qvalue = std::floor(qvalue + static_cast<REAL>(0.5));
                        auto dn_qvalue = std::ceil(qvalue - static_cast<REAL>(0.5));
                        auto rem = std::fmod(up_qvalue, 2.0);
                        qvalue = (rem == 0.0) ? up_qvalue : dn_qvalue;
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_TOWARD_INFINITY)
                    {
                        auto abs_qvalue = std::fabs(qvalue);
                        auto abs_qvalue_toward_inf = std::ceil(abs_qvalue);
                        qvalue = (qvalue < static_cast<REAL>(0.0)) ? -abs_qvalue_toward_inf
                                                                   : abs_qvalue_toward_inf;
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_TOWARD_ZERO)
                    {
                        auto abs_qvalue = std::fabs(qvalue);
                        auto abs_qvalue_toward_zero = std::floor(abs_qvalue);
                        qvalue = (qvalue < static_cast<REAL>(0.0)) ? -abs_qvalue_toward_zero
                                                                   : abs_qvalue_toward_zero;
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_UP)
                    {
                        qvalue = std::ceil(qvalue);
                    }
                    else if (round_mode == op::Quantize::RoundMode::ROUND_DOWN)
                    {
                        qvalue = std::floor(qvalue);
                    }

                    // apply zero_point
                    qvalue += zero_point[scale_zero_point_transform.index(scale_zero_point_coord)];

                    // clamp
                    qvalue = std::max<REAL>(qvalue,
                                            static_cast<REAL>(std::numeric_limits<QUANT>::min()));
                    qvalue = std::min<REAL>(qvalue,
                                            static_cast<REAL>(std::numeric_limits<QUANT>::max()));

                    // cast
                    output[input_transform.index(input_coord)] = static_cast<QUANT>(qvalue);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
