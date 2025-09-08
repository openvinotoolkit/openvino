// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <random>
#include <stdexcept>
#include "openvino/core/type/float16.hpp"

// NOTE: Needed only for possibly imported type (always_false).
#include <type_traits>

namespace rnd_generators
{
    namespace meta
    {
        // NOTE: Should be imported from clDNN API headers.
        template <typename Ty> struct always_false : std::false_type {};
    }

    template <typename NumberTy>
    struct number_caps
    {
        static_assert(meta::always_false<NumberTy>::value, "Capabilities were not defined for selected number type. Please add specialization.");
    };

    template <>
    struct number_caps<float>
    {
        using output_type = float;
        using calc_type = float;
        using rnd_type = std::int32_t;

        static constexpr unsigned significand_bits = 23; // Number of stored bits of significand part of FP32.

        static constexpr calc_type inv_exp2(const unsigned magnitude)
        {
            return magnitude == 0U ? 1.0f : (magnitude & 1U ? 0.5f : 1.0f) * inv_exp2(magnitude >> 1) * inv_exp2(magnitude >> 1);
        }

        static constexpr output_type convert(const calc_type value)
        {
            return value;
        }
    };

    static_assert(number_caps<float>::inv_exp2(0) == 1.0f, "1/exp2(0)");
    static_assert(number_caps<float>::inv_exp2(1) == 0.5f, "1/exp2(1)");
    static_assert(number_caps<float>::inv_exp2(2) == 0.25f, "1/exp2(2)");
    static_assert(number_caps<float>::inv_exp2(3) == 0.125f, "1/exp2(3)");
    static_assert(number_caps<float>::inv_exp2(4) == 0.0625f, "1/exp2(4)");
    static_assert(number_caps<float>::inv_exp2(5) == 0.03125f, "1/exp2(5)");
    static_assert(number_caps<float>::inv_exp2(6) == 0.015625f, "1/exp2(6)");
    static_assert(number_caps<float>::inv_exp2(7) == 0.0078125f, "1/exp2(7)");
    static_assert(number_caps<float>::inv_exp2(8) == 0.00390625f, "1/exp2(8)");

    template <>
    struct number_caps<double>
    {
        using output_type = double;
        using calc_type = double;
        using rnd_type = std::int64_t;

        static constexpr unsigned significand_bits = 52; // Number of stored bits of significand part of FP.

        static constexpr calc_type inv_exp2(const unsigned magnitude)
        {
            return magnitude == 0U ? 1.0 : (magnitude & 1U ? 0.5 : 1.0) * inv_exp2(magnitude >> 1) * inv_exp2(magnitude >> 1);
        }

        static constexpr output_type convert(const calc_type value)
        {
            return value;
        }
    };

    static_assert(number_caps<double>::inv_exp2(0) == 1.0, "1/exp2(0)");
    static_assert(number_caps<double>::inv_exp2(1) == 0.5, "1/exp2(1)");
    static_assert(number_caps<double>::inv_exp2(2) == 0.25, "1/exp2(2)");
    static_assert(number_caps<double>::inv_exp2(3) == 0.125, "1/exp2(3)");
    static_assert(number_caps<double>::inv_exp2(4) == 0.0625, "1/exp2(4)");
    static_assert(number_caps<double>::inv_exp2(5) == 0.03125, "1/exp2(5)");
    static_assert(number_caps<double>::inv_exp2(6) == 0.015625, "1/exp2(6)");
    static_assert(number_caps<double>::inv_exp2(7) == 0.0078125, "1/exp2(7)");
    static_assert(number_caps<double>::inv_exp2(8) == 0.00390625, "1/exp2(8)");

    template <>
    struct number_caps<ov::float16> : number_caps<float>
    {
        using output_type = ov::float16; // NOTE: Exchange with actual ov::float16.

        static constexpr unsigned significand_bits = 10; // Number of stored bits of significand part of FP.

        static output_type convert(const calc_type value)
        {
            return ov::float16(value);
        }
    };

    template <typename NumberTy, typename RndEngineTy>
    auto gen_number(RndEngineTy& rnd_engine,
        const unsigned significand_rnd_bits = number_caps<NumberTy>::significand_bits,
        const bool rnd_sign = true,
        const bool exclude_ones = false,
        const unsigned scale = 1)
        -> typename number_caps<NumberTy>::output_type
    {
        using rnd_type = typename number_caps<NumberTy>::rnd_type;
        using calc_type = typename number_caps<NumberTy>::calc_type;

        constexpr rnd_type rnd_zero = 0;
        constexpr rnd_type rnd_one = 1;

        if (significand_rnd_bits > number_caps<NumberTy>::significand_bits)
            throw std::logic_error("Number of random bits is longer than sigificand part stored in the number.");

        rnd_type rnd_min = rnd_sign ? static_cast<rnd_type>(exclude_ones) - (rnd_one << significand_rnd_bits) : rnd_zero;
        rnd_type rnd_max = (rnd_one << significand_rnd_bits) - static_cast<rnd_type>(exclude_ones);

        calc_type calc_base = number_caps<NumberTy>::inv_exp2(significand_rnd_bits) * static_cast<calc_type>(scale);

        std::uniform_int_distribution<rnd_type> distribution(rnd_min, rnd_max);

        // In loop will be more efficient.
        calc_type rnd_val = static_cast<calc_type>(distribution(rnd_engine));

        return number_caps<NumberTy>::convert(calc_base * rnd_val);
    }

} // namespace rnd_generators
