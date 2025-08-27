// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <immintrin.h>

#include <array>
#include <cstdint>
#include <iomanip>

#define ASSERT_NO_THROW_WITH_MESSAGE(code)   \
    do {                                     \
        try {                                \
            code;                            \
        } catch (const std::exception& ex) { \
            FAIL() << ex.what();             \
        } catch (...) {                      \
            FAIL() << "Unknown exception";   \
        }                                    \
    } while (0)

#define ASSERT_NO_THROW_IF(condition, code)     \
    do {                                        \
        if (condition) {                        \
            ASSERT_NO_THROW_WITH_MESSAGE(code); \
        } else {                                \
            ASSERT_ANY_THROW(code);             \
        }                                       \
    } while (0);

#define Tensors [](std::vector<int> & input)

using ShapesInitializer = std::function<void(std::vector<int>&)>;

namespace details {

uint8_t read_4b(const uint8_t* data, size_t r, size_t c, size_t cols);
void write_4b(uint8_t* data, uint8_t val, size_t r, size_t c, size_t cols);

::testing::AssertionResult ArraysMatch(const std::vector<int8_t>& actual, const std::vector<int8_t>& expected);

std::vector<size_t> compute_strides(const std::vector<size_t>& dims);

std::vector<size_t> reorder(const std::vector<size_t>& dims, const std::vector<size_t>& order);

::testing::internal::ParamGenerator<typename std::vector<ShapesInitializer>::value_type> ShapesIn(
    const std::vector<ShapesInitializer>& container);

}  // namespace details