// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/core/except.hpp"


#define GET_SUITE_NAME  (std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) + \
                         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()))

namespace tests {
static const uint32_t DEFAULT_SEED = 0;


inline uint64_t stable_string_seed(std::string_view seed) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char ch : seed) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return hash;
}


class random_generator {
public:
    random_generator() = default;

    random_generator(const std::string& seed) {
        set_seed(seed);
    }

    std::default_random_engine& get_generator() {
        return generator;
    }

    void set_seed(const std::string& seed) {
        const auto seed_hash = stable_string_seed(seed);
        set_seed(seed_hash);
    }

    void set_seed(const uint32_t seed) {
        generator.seed(seed);
    }

    void set_seed(const uint64_t seed) {
        const uint32_t seed_lo = static_cast<uint32_t>(seed);
        const uint32_t seed_hi = static_cast<uint32_t>(seed >> 32);
        std::seed_seq seed_seq{seed_lo, seed_hi};
        generator.seed(seed_seq);
    }

    template<typename ReturnType>
    ReturnType generate_random_val(double min, double max, int k = 8) {
        // 1/k is the resolution of the floating point numbers
        auto distribution = make_distribution(min, max, k);
        ReturnType val = static_cast<ReturnType>(distribution(this->generator));
        val /= k;

        return val;
    }

    template<typename ReturnType>
    std::vector<ReturnType> generate_random_1d(size_t a, double min, double max, int k = 8) {
        // 1/k is the resolution of the floating point numbers
        auto distribution = make_distribution(min, max, k);
        std::vector<ReturnType> v(a);

        for (size_t i = 0; i < a; ++i) {
            v[i] = static_cast<ReturnType>(distribution(this->generator));
            v[i] /= k;
        }
        return v;
    }

    template<typename ReturnType>
    std::vector<std::vector<ReturnType>> generate_random_2d(size_t a, size_t b, double min, double max, int k = 8) {
        std::vector<std::vector<ReturnType>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_1d<ReturnType>(b, min, max, k);
        return v;
    }

    template<typename ReturnType>
    std::vector<std::vector<std::vector<ReturnType>>> generate_random_3d(size_t a, size_t b, size_t c, double min, double max, int k = 8) {
        std::vector<std::vector<std::vector<ReturnType>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_2d<ReturnType>(b, c, min, max, k);
        return v;
    }

    // parameters order is assumed to be bfyx or bfyx
    template<typename ReturnType>
    std::vector<std::vector<std::vector<std::vector<ReturnType>>>> generate_random_4d(size_t a, size_t b, size_t c, size_t d,
                                                                                     double min, double max, int k = 8) {
        std::vector<std::vector<std::vector<std::vector<ReturnType>>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_3d<ReturnType>(b, c, d, min, max, k);
        return v;
    }

    // parameters order is assumed to be sbfyx for filters when split > 1
    template<typename ReturnType>
    std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>> generate_random_5d(size_t a, size_t b, size_t c, size_t d, size_t e,
                                                                                                   double min, double max, int k = 8) {
        std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_4d<ReturnType>(b, c, d, e, min, max, k);
        return v;
    }

    template<typename ReturnType>
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>>> generate_random_6d(size_t a, size_t b, size_t c, size_t d,
                                                                                                    size_t e, size_t f, double min, double max, int k = 8) {
        std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_5d<ReturnType>(b, c, d, e, f, min, max, k);
        return v;
    }

    template<typename ReturnType>
    std::vector<ReturnType> generate_random_norepetitions(size_t size, int min, int max, float bound = 0.45) {
        // Rerurn repeatless vector with size = size in range(min, max)
        std::uniform_int_distribution<int> distribution(min, max);
        std::uniform_real_distribution<float> to_bound_dist(0, bound);
        std::set<int> repeatless;
        std::vector<float> v(size, 0);
        std::vector<ReturnType> res(size);
        int i = 0;
        int temp;
        if (max - min >= static_cast<int>(size) - 1) {
            while (repeatless.size() < size) {
                temp = distribution(this->generator);
                if (repeatless.find(temp) == repeatless.end()) {
                    repeatless.insert(temp);
                    v[i] = static_cast<float>(temp);
                    i++;
                }
            }
            for (size_t k = 0; k < v.size(); k++) {
                v[k] += to_bound_dist(this->generator);
                res[k] = static_cast<ReturnType>(v[k]);
            }
        } else {
            throw "Array size is bigger than size of range(min, max). Unable to generate array of unique integer numbers";
        }
        return res;
    }

private:
    // Values are drawn on a 1/k grid, so the integer bounds are k*min and k*max.
    // min/max are taken as floating point on purpose: they used to be int, which silently truncated
    // calls such as generate_random_1d<ov::float16>(n, -0.25f, 0.25f) into a constant-zero range.
    static std::uniform_int_distribution<int> make_distribution(double min, double max, int k) {
        OPENVINO_ASSERT(min <= max, "random_generator: min (", min, ") must not exceed max (", max, ")");
        OPENVINO_ASSERT(k > 0, "random_generator: resolution k must be positive, got ", k);

        if (min == max) {
            const auto val = static_cast<int>(std::lround(min * k));
            return std::uniform_int_distribution<int>(val, val);
        }

        // ceil/floor (rather than rounding) keeps every generated value inside [min, max]
        const auto lo = static_cast<int>(std::ceil(min * k));
        const auto hi = static_cast<int>(std::floor(max * k));
        OPENVINO_ASSERT(lo < hi,
                        "random_generator: range [", min, ", ", max, "] holds fewer than 2 values at resolution 1/", k,
                        ", so the data would be constant. Pass k >= ", static_cast<int>(std::ceil(2.0 / (max - min))));

        return std::uniform_int_distribution<int>(lo, hi);
    }

    std::default_random_engine generator{DEFAULT_SEED};
};

} // namespace tests
