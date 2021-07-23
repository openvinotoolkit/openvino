// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ctime>
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            std::pair<uint32_t, uint32_t> split_high_low(uint64_t value)
            {
                uint32_t low = static_cast<uint32_t>(value);
                uint32_t high = static_cast<uint32_t>(value >> 32);
                return {low, high};
            }

            uint64_t unite_high_low(uint32_t high, uint32_t low)
            {
                return (static_cast<uint64_t>(high) << 32) + low;
            }

            void calculate_round(const uint64_t& key, uint64_t& counter, uint64_t& n)
            {
                auto counter_lr = split_high_low(counter);
                auto key_lr = split_high_low(key);
                auto n_lr = split_high_low(n);
                auto prod0 = split_high_low(static_cast<uint64_t>(0xD2511F53) * n_lr.first);
                auto prod1 = split_high_low(static_cast<uint64_t>(0xCD9E8D57) * counter_lr.first);

                n_lr.first = prod1.second ^ n_lr.second ^ key_lr.first;
                n_lr.second = prod1.first;
                counter_lr.first = prod0.second ^ counter_lr.second ^ key_lr.second;
                counter_lr.second = prod0.first;

                counter = unite_high_low(counter_lr.second, counter_lr.first);
                n = unite_high_low(n_lr.second, n_lr.first);
            }

            void raise_key(uint64_t& key)
            {
                auto key_lr = split_high_low(key);
                key_lr.first += 0x9E3779B9;
                key_lr.second += 0xBB67AE85;
                key = unite_high_low(key_lr.second, key_lr.first);
            }

            float uint32_to_float(uint32_t x)
            {
                uint32_t x_uint32 = (static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu);

                float x_float;
                memcpy(&x_float, &x_uint32, sizeof(x_uint32));
                return x_float - 1.0f;
            }

            float16 uint32_to_float16(uint32_t x)
            {
                uint16_t x_uint16 = static_cast<uint16_t>(x);
                x_uint16 = (static_cast<uint16_t>(15) << 10) | (x_uint16 & 0x3ffu);

                float16 x_float16;
                memcpy(&x_float16, &x_uint16, sizeof(x_uint16));
                return x_float16 - static_cast<float16>(1);
            }

            double uint32_to_double(uint32_t x1, uint32_t x2)
            {
                uint64_t mantissa =
                    ((static_cast<uint64_t>(x1) & 0xfffffu) << 32) | static_cast<uint64_t>(x2);
                uint64_t x_uint64 = ((static_cast<uint64_t>(1023) << 52) | mantissa);

                double x_double;
                memcpy(&x_double, &x_uint64, sizeof(x_uint64));
                return x_double - 1.0;
            }

            uint64_t uint32_to_uint64(uint32_t x1, uint32_t x2)
            {
                return (static_cast<uint64_t>(x2) << 32) | static_cast<uint64_t>(x1);
            }

            bfloat16 uint32_to_bfloat16(uint32_t x)
            {
                uint16_t x_uint16 = static_cast<uint16_t>(x);
                x_uint16 = (static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu);

                bfloat16 x_bfloat16;
                memcpy(&x_bfloat16, &x_uint16, sizeof(x_uint16));
                return x_bfloat16 - static_cast<bfloat16>(1);
            }

            void run_philox(uint64_t key,
                            uint64_t counter,
                            uint64_t n,
                            size_t n_rounds,
                            std::vector<uint32_t>& res)
            {
                for (size_t i = 0; i < n_rounds; i++)
                {
                    calculate_round(key, counter, n);
                    if (i < n_rounds - 1)
                        raise_key(key);
                }
                auto res1 = split_high_low(n);
                auto res2 = split_high_low(counter);
                res[0] = res1.first;
                res[1] = res1.second;
                res[2] = res2.first;
                res[3] = res2.second;
            }

            void random_uniform(const uint64_t* out_shape,
                                const char* min_val,
                                const char* max_val,
                                char* out,
                                const Shape& out_shape_shape,
                                ngraph::element::Type elem_type,
                                uint64_t seed,
                                uint64_t seed2)
            {
                if (seed == 0 && seed2 == 0)
                {
                    std::srand(std::time(nullptr));
                    seed = std::rand();
                }
                uint64_t key = seed;
                uint64_t counter = seed2;
                uint64_t n = 0;
                size_t shape_count = shape_size(out_shape_shape);
                size_t elem_count = 1;
                for (size_t i = 0; i < shape_count; i++)
                {
                    elem_count *= out_shape[i];
                }
                size_t step = elem_type.size() > 4 ? 2 : 4;
                for (size_t k = 0; k < elem_count; k += step)
                {
                    std::vector<uint32_t> res(4);
                    run_philox(key, counter, n, 10, res);
                    switch (elem_type)
                    {
                    case ngraph::element::Type_t::f32:
                    {
                        float res_float[4];
                        std::transform(res.data(), res.data() + 4, res_float, uint32_to_float);
                        float mn[1];
                        float mx[1];
                        memcpy(mn, min_val, elem_type.size());
                        memcpy(mx, max_val, elem_type.size());
                        std::transform(res.data(),
                                       res.data() + 4,
                                       res_float,
                                       [&mn, &mx](const uint32_t& elem) {
                                           return uint32_to_float(elem) * (mx[0] - mn[0]) + mn[0];
                                       });

                        memcpy(out + k * elem_type.size(),
                               res_float,
                               std::min((size_t)4, elem_count - k) * elem_type.size());
                        break;
                    }
                    case ngraph::element::Type_t::f16:
                    {
                        float16 res_float16[4];
                        std::transform(res.data(), res.data() + 4, res_float16, uint32_to_float16);
                        float16 mn[1];
                        float16 mx[1];
                        memcpy(mn, min_val, elem_type.size());
                        memcpy(mx, max_val, elem_type.size());
                        std::transform(res.data(),
                                       res.data() + 4,
                                       res_float16,
                                       [&mn, &mx](const uint32_t& elem) {
                                           return uint32_to_float16(elem) * (mx[0] - mn[0]) + mn[0];
                                       });
                        memcpy(out + k * elem_type.size(),
                               res_float16,
                               std::min((size_t)4, elem_count - k) * elem_type.size());
                        break;
                    }
                    case ngraph::element::Type_t::bf16:
                    {
                        bfloat16 res_bfloat16[4];
                        bfloat16 mn[1];
                        bfloat16 mx[1];
                        memcpy(mn, min_val, elem_type.size());
                        memcpy(mx, max_val, elem_type.size());
                        std::transform(res.data(),
                                       res.data() + 4,
                                       res_bfloat16,
                                       [&mn, &mx](const uint32_t& elem) {
                                           return uint32_to_bfloat16(elem) * (mx[0] - mn[0]) +
                                                  mn[0];
                                       });
                        memcpy(out + k * elem_type.size(),
                               res_bfloat16,
                               std::min((size_t)4, elem_count - k) * elem_type.size());
                        break;
                    }
                    case ngraph::element::Type_t::f64:
                    {
                        double res_double[2];
                        res_double[0] = uint32_to_double(res[0], res[1]);
                        res_double[1] = uint32_to_double(res[2], res[3]);
                        double mn[1];
                        double mx[1];
                        memcpy(mn, min_val, elem_type.size());
                        memcpy(mx, max_val, elem_type.size());
                        res_double[0] = uint32_to_double(res[0], res[1]) * (mx[0] - mn[0]) + mn[0];
                        res_double[1] = uint32_to_double(res[2], res[3]) * (mx[0] - mn[0]) + mn[0];
                        memcpy(out + k * elem_type.size(),
                               res_double,
                               std::min((size_t)2, elem_count - k) * elem_type.size());
                        break;
                    }
                    case ngraph::element::Type_t::i32:
                    {
                        int res_int[4];
                        int mn[1];
                        int mx[1];
                        memcpy(mn, min_val, elem_type.size());
                        memcpy(mx, max_val, elem_type.size());
                        std::transform(
                            res.data(), res.data() + 4, res_int, [&mn, &mx](const uint32_t& elem) {
                                return elem % (mx[0] - mn[0]) + mn[0];
                            });
                        memcpy(out + k * elem_type.size(),
                               res_int,
                               std::min((size_t)4, elem_count - k) * elem_type.size());
                        break;
                    }
                    case ngraph::element::Type_t::i64:
                    {
                        int64_t res_int64[2];
                        int64_t mn[1];
                        int64_t mx[1];
                        memcpy(mn, min_val, elem_type.size());
                        memcpy(mx, max_val, elem_type.size());
                        res_int64[0] = uint32_to_uint64(res[0], res[1]) % (mx[0] - mn[0]) + mn[0];
                        res_int64[1] = uint32_to_uint64(res[2], res[3]) % (mx[0] - mn[0]) + mn[0];
                        memcpy(out + k * elem_type.size(),
                               res_int64,
                               std::min((size_t)2, elem_count - k) * elem_type.size());
                        break;
                    }
                    default:
                        throw ngraph_error("Unsupported type of RandomUniform: " +
                                           elem_type.get_type_name());
                    }
                    if (++n == 0)
                        ++counter;
                }
            }

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
