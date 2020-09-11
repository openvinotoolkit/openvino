//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test_tools.hpp"

static constexpr int BFLOAT_MANTISSA_BITS = 8;
static constexpr int FLOAT_MANTISSA_BITS = 24;
static constexpr int DOUBLE_MANTISSA_BITS = 53;

// Maximum available float bits
#ifndef MAX_FLOAT_BITS
#define MAX_FLOAT_BITS FLOAT_MANTISSA_BITS
#endif

// Minimum float tolerance bits possible
#ifndef MIN_FLOAT_TOLERANCE_BITS
#define MIN_FLOAT_TOLERANCE_BITS (FLOAT_MANTISSA_BITS - MAX_FLOAT_BITS)
#endif

static_assert((MAX_FLOAT_BITS > 0) && (MAX_FLOAT_BITS <= FLOAT_MANTISSA_BITS),
              "MAX_FLOAT_BITS must be in range (0, 24]");
static_assert((MIN_FLOAT_TOLERANCE_BITS >= 0) && (MIN_FLOAT_TOLERANCE_BITS < FLOAT_MANTISSA_BITS),
              "MIN_FLOAT_TOLERANCE_BITS must be in range [0, 24)");

// Default float tolerance bits
#ifndef DEFAULT_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS (MIN_FLOAT_TOLERANCE_BITS + 2)
#endif

// Default float tolerance bits
#ifndef DEFAULT_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS 2
#endif

static_assert((DEFAULT_FLOAT_TOLERANCE_BITS >= 0) &&
                  (DEFAULT_FLOAT_TOLERANCE_BITS < FLOAT_MANTISSA_BITS),
              "DEFAULT_FLOAT_TOLERANCE_BITS must be in range [0, 24)");

static_assert((DEFAULT_DOUBLE_TOLERANCE_BITS >= 0) &&
                  (DEFAULT_DOUBLE_TOLERANCE_BITS < DOUBLE_MANTISSA_BITS),
              "DEFAULT_DOUBLE_TOLERANCE_BITS must be in range [0, 53)");

namespace ngraph
{
    namespace test
    {
        // clang-format off
        /// \brief Determine distance between two f32 numbers
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param min_signal Minimum value for comparisons
        /// \returns Distance
        ///
        /// References:
        /// - https://en.wikipedia.org/wiki/Unit_in_the_last_place
        /// - https://randomascii.wordpress.com/2012/01/23/stupid-float-tricks-2
        /// - https://github.com/google/googletest/blob/master/googletest/docs/AdvancedGuide.md#floating-point-comparison
        ///
        /// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
        /// |------------bfloat-----------|
        /// |----------------------------float----------------------------|
        ///
        /// bfloat (s1, e8, m7) has 7 + 1 = 8 bits of mantissa or bit_precision
        /// float (s1, e8, m23) has 23 + 1 = 24 bits of mantissa or bit_precision
        ///
        /// This function uses hard-coded value of 8 bit exponent_bits, so it's only valid for
        /// bfloat and f32.
        // clang-format on
        uint32_t float_distance(float a, float b, float min_signal = 0.0f);

        // clang-format off
        /// \brief Determine distance between two f64 numbers
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param min_signal Minimum value for comparisons
        /// \returns Distance
        ///
        /// References:
        /// - https://en.wikipedia.org/wiki/Unit_in_the_last_place
        /// - https://randomascii.wordpress.com/2012/01/23/stupid-float-tricks-2
        /// - https://github.com/google/googletest/blob/master/googletest/docs/AdvancedGuide.md#floating-point-comparison
        ///
        /// s e e e e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m
        /// |----------------------------double-------------------------------------------------------------------------------------------|
        ///
        /// double (s1, e11, m52) has 52 + 1 = 53 bits of mantissa or bit_precision
        ///
        /// This function uses hard-coded value of 11 bit exponent_bits, so it's only valid for f64.
        // clang-format on
        uint64_t float_distance(double a, double b, double min_signal = 0.0);

        // clang-format off
        /// \brief Check if the two f32 numbers are close
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param tolerance_bits Bit tolerance error
        /// \param min_signal Minimum value for comparisons
        /// \returns True iff the distance between a and b is within 2 ^ tolerance_bits ULP
        ///
        /// References:
        /// - https://en.wikipedia.org/wiki/Unit_in_the_last_place
        /// - https://randomascii.wordpress.com/2012/01/23/stupid-float-tricks-2
        /// - https://github.com/abseil/googletest/blob/master/googletest/docs/advanced.md#floating-point-comparison
        ///
        /// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
        /// |------------bfloat-----------|
        /// |----------------------------float----------------------------|
        ///
        /// bfloat (s1, e8, m7) has 7 + 1 = 8 bits of mantissa or bit_precision
        /// float (s1, e8, m23) has 23 + 1 = 24 bits of mantissa or bit_precision
        ///
        /// This function uses hard-coded value of 8 bit exponent_bits, so it's only valid for
        /// bfloat and f32.
        // clang-format on
        bool close_f(float a,
                     float b,
                     int tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS,
                     float min_signal = 0.0f);

        // clang-format off
        /// \brief Check if the two f64 numbers are close
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param tolerance_bits Bit tolerance error
        /// \param min_signal Minimum value for comparisons
        /// \returns True iff the distance between a and b is within 2 ^ tolerance_bits ULP
        ///
        /// References:
        /// - https://en.wikipedia.org/wiki/Unit_in_the_last_place
        /// - https://randomascii.wordpress.com/2012/01/23/stupid-float-tricks-2
        /// - https://github.com/abseil/googletest/blob/master/googletest/docs/advanced.md#floating-point-comparison
        ///
        /// s e e e e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m m
        /// |----------------------------double-------------------------------------------------------------------------------------------|
        ///
        /// double (s1, e11, m52) has 52 + 1 = 53 bits of mantissa or bit_precision
        ///
        /// This function uses hard-coded value of 11 bit exponent_bits, so it's only valid for f64.
        // clang-format on
        bool close_f(double a,
                     double b,
                     int tolerance_bits = DEFAULT_DOUBLE_TOLERANCE_BITS,
                     double min_signal = 0.0);

        /// \brief Determine distances between two vectors of f32 numbers
        /// \param a Vector of floats to compare
        /// \param b Vector of floats to compare
        /// \param min_signal Minimum value for comparisons
        /// \returns Vector of distances
        ///
        /// See float_distance for limitations and assumptions.
        std::vector<uint32_t> float_distances(const std::vector<float>& a,
                                              const std::vector<float>& b,
                                              float min_signal = 0.0f);

        /// \brief Determine distances between two vectors of f64 numbers
        /// \param a Vector of doubles to compare
        /// \param b Vector of doubles to compare
        /// \param min_signal Minimum value for comparisons
        /// \returns Vector of distances
        ///
        /// See float_distance for limitations and assumptions.
        std::vector<uint64_t> float_distances(const std::vector<double>& a,
                                              const std::vector<double>& b,
                                              double min_signal = 0.0);

        /// \brief Determine number of matching mantissa bits given a distance
        /// \param distance Distance calculated by float_distance
        /// \returns Number of matching mantissa bits
        ///
        /// See float_distance for limitations and assumptions.
        uint32_t matching_mantissa_bits(uint32_t distance);

        /// \brief Determine number of matching mantissa bits given a distance
        /// \param distance Distance calculated by float_distance
        /// \returns Number of matching mantissa bits
        ///
        /// See float_distance for limitations and assumptions.
        uint32_t matching_mantissa_bits(uint64_t distance);

        /// \brief Check if the two floating point vectors are all close
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param tolerance_bits Bit tolerance error
        /// \param min_signal Minimum value for comparisons
        /// \returns ::testing::AssertionSuccess iff the two floating point vectors are close
        ::testing::AssertionResult all_close_f(const std::vector<float>& a,
                                               const std::vector<float>& b,
                                               int tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS,
                                               float min_signal = 0.0f);

        /// \brief Check if the two double floating point vectors are all close
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param tolerance_bits Bit tolerance error
        /// \param min_signal Minimum value for comparisons
        /// \returns ::testing::AssertionSuccess iff the two floating point vectors are close
        ::testing::AssertionResult all_close_f(const std::vector<double>& a,
                                               const std::vector<double>& b,
                                               int tolerance_bits = DEFAULT_DOUBLE_TOLERANCE_BITS,
                                               double min_signal = 0.0);

        /// \brief Check if the two TensorViews are all close in float
        /// \param a First Tensor to compare
        /// \param b Second Tensor to compare
        /// \param tolerance_bits Bit tolerance error
        /// \param min_signal Minimum value for comparisons
        /// Returns true iff the two TensorViews are all close in float
        ::testing::AssertionResult all_close_f(const std::shared_ptr<runtime::Tensor>& a,
                                               const std::shared_ptr<runtime::Tensor>& b,
                                               int tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS,
                                               float min_signal = 0.0f);

        /// \brief Check if the two vectors of TensorViews are all close in float
        /// \param as First vector of Tensor to compare
        /// \param bs Second vector of Tensor to compare
        /// \param tolerance_bits Bit tolerance error
        /// \param min_signal Minimum value for comparisons
        /// Returns true iff the two TensorViews are all close in float
        ::testing::AssertionResult
            all_close_f(const std::vector<std::shared_ptr<runtime::Tensor>>& as,
                        const std::vector<std::shared_ptr<runtime::Tensor>>& bs,
                        int tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS,
                        float min_signal = 0.0f);
    }
}
