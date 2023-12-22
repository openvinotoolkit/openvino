// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>
#include <set>

#define GET_SUITE_NAME  (std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) + \
                         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()))

namespace tests {
static const uint32_t DEFAULT_SEED = 0;

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
        auto seed_hash = std::hash<std::string>{}(seed);
        set_seed(static_cast<uint32_t>(seed_hash));
    }

    void set_seed(const uint32_t seed) {
        generator = std::default_random_engine{seed};
    }

    template<typename ReturnType>
    ReturnType generate_random_val(int min, int max, int k = 8) {
        // 1/k is the resolution of the floating point numbers
        std::uniform_int_distribution<int> distribution(k * min, k * max);
        ReturnType val = static_cast<ReturnType>(distribution(this->generator));
        val /= k;

        return val;
    }

    template<typename ReturnType>
    std::vector<ReturnType> generate_random_1d(size_t a, int min, int max, int k = 8) {
        // 1/k is the resolution of the floating point numbers
        std::uniform_int_distribution<int> distribution(k * min, k * max);
        std::vector<ReturnType> v(a);

        for (size_t i = 0; i < a; ++i) {
            v[i] = static_cast<ReturnType>(distribution(this->generator));
            v[i] /= k;
        }
        return v;
    }

    template<typename ReturnType>
    std::vector<std::vector<ReturnType>> generate_random_2d(size_t a, size_t b, int min, int max, int k = 8) {
        std::vector<std::vector<ReturnType>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_1d<ReturnType>(b, min, max, k);
        return v;
    }

    template<typename ReturnType>
    std::vector<std::vector<std::vector<ReturnType>>> generate_random_3d(size_t a, size_t b, size_t c, int min, int max, int k = 8) {
        std::vector<std::vector<std::vector<ReturnType>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_2d<ReturnType>(b, c, min, max, k);
        return v;
    }

    // parameters order is assumed to be bfyx or bfyx
    template<typename ReturnType>
    std::vector<std::vector<std::vector<std::vector<ReturnType>>>> generate_random_4d(size_t a, size_t b, size_t c, size_t d, int min, int max, int k = 8) {
        std::vector<std::vector<std::vector<std::vector<ReturnType>>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_3d<ReturnType>(b, c, d, min, max, k);
        return v;
    }

    // parameters order is assumed to be sbfyx for filters when split > 1
    template<typename ReturnType>
    std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>> generate_random_5d(size_t a, size_t b, size_t c, size_t d, size_t e,
                                                                                                   int min, int max, int k = 8) {
        std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>> v(a);
        for (size_t i = 0; i < a; ++i)
            v[i] = generate_random_4d<ReturnType>(b, c, d, e, min, max, k);
        return v;
    }

    template<typename ReturnType>
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<ReturnType>>>>>> generate_random_6d(size_t a, size_t b, size_t c, size_t d,
                                                                                                    size_t e, size_t f, int min, int max, int k = 8) {
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
    std::default_random_engine generator{DEFAULT_SEED};
};

} // namespace tests
