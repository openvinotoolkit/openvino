// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/all_close_f.hpp"

#include "common_test_utils/float_util.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "precomp.hpp"

namespace ov {
namespace test {
namespace utils {

constexpr uint32_t FLOAT_BELOW_MIN_SIGNAL = UINT_MAX;
constexpr uint32_t FLOAT_MAX_DIFF = UINT_MAX - 1;
constexpr uint64_t DOUBLE_BELOW_MIN_SIGNAL = ULLONG_MAX;
constexpr uint64_t DOUBLE_MAX_DIFF = ULLONG_MAX - 1;

uint32_t float_distance(float a, float b, float min_signal) {
    if (std::isnan(a) && std::isnan(b)) {
        return 0;
    } else if (std::isinf(a) && std::isinf(b)) {
        if (a > 0 && b > 0) {
            return 0;
        } else if (a < 0 && b < 0) {
            return 0;
        }
        return FLOAT_MAX_DIFF;
    }

    FloatUnion a_fu(a);
    FloatUnion b_fu(b);
    FloatUnion min_signal_fu(min_signal);
    uint32_t a_uint = a_fu.i;
    uint32_t b_uint = b_fu.i;

    // A trick to handle both positive and negative numbers, see https://goo.gl/YbdnFQ
    // - If negative: convert to two's complement
    // - If positive: mask with sign bit
    uint32_t sign_mask = static_cast<uint32_t>(1U) << 31;
    uint32_t abs_value_bits_mask = ~sign_mask;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint32_t distance;
    uint32_t a_uint_abs = (abs_value_bits_mask & a_fu.i);
    uint32_t b_uint_abs = (abs_value_bits_mask & b_fu.i);
    uint32_t min_signal_uint_abs = (abs_value_bits_mask & min_signal_fu.i);
    if ((a_uint_abs < min_signal_uint_abs) && (b_uint_abs < min_signal_uint_abs)) {
        // Both a & b below minimum signal
        distance = FLOAT_BELOW_MIN_SIGNAL;
    } else {
        distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);
        // We've reserved UINT_MAX to mean FLOAT_BELOW_MIN_SIGNAL
        if (distance == UINT_MAX) {
            distance = FLOAT_MAX_DIFF;
        }
    }

    return distance;
}

uint64_t float_distance(double a, double b, double min_signal) {
    if (std::isnan(a) && std::isnan(b)) {
        return 0;
    } else if (std::isinf(a) && std::isinf(b)) {
        if (a > 0 && b > 0) {
            return 0;
        } else if (a < 0 && b < 0) {
            return 0;
        }
        return DOUBLE_MAX_DIFF;
    }

    DoubleUnion a_du(a);
    DoubleUnion b_du(b);
    DoubleUnion min_signal_du(min_signal);
    uint64_t a_uint = a_du.i;
    uint64_t b_uint = b_du.i;

    // A trick to handle both positive and negative numbers, see https://goo.gl/YbdnFQ
    // - If negative: convert to two's complement
    // - If positive: mask with sign bit
    uint64_t sign_mask = static_cast<uint64_t>(1U) << 63;
    uint64_t abs_value_bits_mask = ~sign_mask;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint64_t distance;
    uint64_t a_uint_abs = (abs_value_bits_mask & a_du.i);
    uint64_t b_uint_abs = (abs_value_bits_mask & b_du.i);
    uint64_t min_signal_uint_abs = (abs_value_bits_mask & min_signal_du.i);
    if ((a_uint_abs < min_signal_uint_abs) && (b_uint_abs < min_signal_uint_abs)) {
        // Both a & b below minimum signal
        distance = DOUBLE_BELOW_MIN_SIGNAL;
    } else {
        distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);
        // We've reserved ULLONG_MAX to mean DOUBLE_BELOW_MIN_SIGNAL
        if (distance == ULLONG_MAX) {
            distance = DOUBLE_MAX_DIFF;
        }
    }

    return distance;
}

bool close_f(float a, float b, int tolerance_bits, float min_signal) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    } else if (std::isinf(a) && std::isinf(b)) {
        if (a > 0 && b > 0) {
            return true;
        } else if (a < 0 && b < 0) {
            return true;
        }
        return false;
    }

    uint32_t distance = float_distance(a, b, min_signal);

    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (FLOAT_MANTISSA_BITS - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;

    return (distance <= tolerance) || (distance == FLOAT_BELOW_MIN_SIGNAL);
}

bool close_f(double a, double b, int tolerance_bits, double min_signal) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    } else if (std::isinf(a) && std::isinf(b)) {
        if (a > 0 && b > 0) {
            return true;
        } else if (a < 0 && b < 0) {
            return true;
        }
        return false;
    }

    uint64_t distance = float_distance(a, b, min_signal);

    // e.g. for double with 52 bit mantissa, 2 bit accuracy, and hard-coded 11 bit exponent_bits
    // tolerance_bit_shift = 64 -           (1 +  11 + (53 -     1         ) - 2             )
    //                       double_length   sign exp   mantissa implicit 1    tolerance_bits
    uint64_t tolerance_bit_shift = 64 - (1 + 11 + (DOUBLE_MANTISSA_BITS - 1) - tolerance_bits);
    uint64_t tolerance = static_cast<uint64_t>(1U) << tolerance_bit_shift;

    return (distance <= tolerance) || (distance == DOUBLE_BELOW_MIN_SIGNAL);
}

std::vector<uint32_t> float_distances(const float* const a, const float* const b, size_t size, float min_signal) {
    std::vector<uint32_t> distances(size);
    for (size_t i = 0; i < size; ++i) {
        distances[i] = float_distance(a[i], b[i], min_signal);
    }

    return distances;
}

std::vector<uint64_t> float_distances(const double* const a, const double* const b, size_t size, double min_signal) {
    std::vector<uint64_t> distances(size);
    for (size_t i = 0; i < size; ++i) {
        distances[i] = float_distance(a[i], b[i], min_signal);
    }

    return distances;
}

uint32_t matching_mantissa_bits(uint32_t distance) {
    uint32_t tolerance_bit_shift = 0;
    uint32_t num_bits_on = 0;

    // Do some bit probing to find the most significant bit that's on,
    // as well as how many bits are on.
    for (uint32_t check_bit = 0; check_bit < 32; ++check_bit) {
        if (distance & (1 << check_bit)) {
            tolerance_bit_shift = check_bit;
            ++num_bits_on;
        }
    }

    // all_close_f is <= test for tolerance (where tolerance is uint32_t with single bit on)
    // So if more than one bit is on we need the next higher tolerance
    if (num_bits_on > 1) {
        ++tolerance_bit_shift;
    }

    // clang-format off
    // all_close_f calculation of tolerance_bit_shift:
    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    //  tolerance_bit_shift   =     32 -          (1 +  8 + (24 -                    1         ) - 2             )
    //                              float_length   sign exp  matching_matissa_bits   implicit 1    tolerance_bits
    //
    // Assuming 0 tolerance_bits and solving for matching_matissa_bits yields:
    //  tolerance_bit_shift   =     32 -          (1 +  8 + (matching_matissa_bits - 1         ) - 0             )
    //  tolerance_bit_shift   =     32 -          (1 +  8 + (matching_matissa_bits - 1         )                 )
    //  matching_matissa_bits =     32 -          (1 +  8 + (tolerance_bit_shift   - 1         )                 )
    // clang-format on
    uint32_t matching_matissa_bits = tolerance_bit_shift < 24 ? (32 - (1 + 8 + (tolerance_bit_shift - 1))) : 0;
    return matching_matissa_bits;
}

uint32_t matching_mantissa_bits(uint64_t distance) {
    uint32_t tolerance_bit_shift = 0;
    uint32_t num_bits_on = 0;

    // Do some bit probing to find the most significant bit that's on,
    // as well as how many bits are on.
    for (uint32_t check_bit = 0; check_bit < 64; ++check_bit) {
        if (distance & (1ull << check_bit)) {
            tolerance_bit_shift = check_bit;
            ++num_bits_on;
        }
    }

    // all_close_f is <= test for tolerance (where tolerance is uint64_t with single bit on)
    // So if more than one bit is on we need the next higher tolerance
    if (num_bits_on > 1) {
        ++tolerance_bit_shift;
    }

    // clang-format off
    // all_close_f calculation of tolerance_bit_shift:
    // e.g. for double with 53 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    //  tolerance_bit_shift   =     64 -          (1 +  11 + (53 -                    1         ) - 2             )
    //                              double_length  sign exp   matching_matissa_bits   implicit 1    tolerance_bits
    //
    // Assuming 0 tolerance_bits and solving for matching_matissa_bits yields:
    //  tolerance_bit_shift   =     64 -          (1 +  11 + (matching_matissa_bits - 1         ) - 0             )
    //  tolerance_bit_shift   =     64 -          (1 +  11 + (matching_matissa_bits - 1         )                 )
    //  matching_matissa_bits =     64 -          (1 +  11 + (tolerance_bit_shift   - 1         )                 )
    // clang-format on
    uint32_t matching_matissa_bits = tolerance_bit_shift < 53 ? (64 - (1 + 11 + (tolerance_bit_shift - 1))) : 0;
    return matching_matissa_bits;
}

::testing::AssertionResult all_close_f(const float* const a,
                                       const float* const b,
                                       size_t size,
                                       int tolerance_bits,
                                       float min_signal) {
    if (tolerance_bits < MIN_FLOAT_TOLERANCE_BITS) {
        tolerance_bits = MIN_FLOAT_TOLERANCE_BITS;
    }
    if (tolerance_bits >= FLOAT_MANTISSA_BITS) {
        tolerance_bits = FLOAT_MANTISSA_BITS - 1;
    }

    bool rc = true;
    std::stringstream msg;
    std::vector<uint32_t> distances = float_distances(a, b, size, min_signal);

    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (FLOAT_MANTISSA_BITS - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;
    uint32_t max_distance = 0;
    uint32_t min_distance = FLOAT_BELOW_MIN_SIGNAL;
    size_t max_distance_index = 0;
    size_t min_distance_index = 0;
    size_t diff_count = 0;
    size_t below_min_count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (distances[i] == FLOAT_BELOW_MIN_SIGNAL) {
            // Special value that indicates both values were below min_signal
            below_min_count++;
            continue;
        }

        if (distances[i] > max_distance) {
            max_distance = distances[i];
            max_distance_index = i;
        }
        if (distances[i] < min_distance) {
            min_distance = distances[i];
            min_distance_index = i;
        }
        bool is_close_f = distances[i] <= tolerance;
        if (!is_close_f) {
            if (diff_count < 5) {
                msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << a[i] << " is not close to "
                    << b[i] << " at index " << i << std::endl;
            }

            rc = false;
            diff_count++;
        }
    }
    if (!rc) {
        msg << "diff count: " << diff_count << " out of " << size << std::endl;
    }
    // Find median value via partial sorting
    size_t middle = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + middle, distances.end());
    uint32_t median_distance = distances[middle];
    if (distances.size() % 2 == 0) {
        // Find middle-1 value
        uint64_t median_sum =
            static_cast<uint64_t>(median_distance) + *max_element(distances.begin(), distances.begin() + middle);
        median_distance = median_sum / 2;
    }

    bool all_below_min_signal = below_min_count == distances.size();
    if (rc) {
        // Short unobtrusive message when passing
        std::cout << "[   INFO   ] Verifying match of <= " << (FLOAT_MANTISSA_BITS - tolerance_bits)
                  << " mantissa bits (" << FLOAT_MANTISSA_BITS << " bits precision - " << tolerance_bits
                  << " tolerance). ";
        if (all_below_min_signal) {
            std::cout << "All values below min_signal: " << min_signal << std::endl;
        } else {
            std::cout << below_min_count << " value(s) below min_signal: " << min_signal << " Loosest match found is "
                      << matching_mantissa_bits(max_distance) << " mantissa bits.\n";
        }
    }

    msg << "passing criteria - mismatch allowed  @ mantissa bit: " << (FLOAT_MANTISSA_BITS - tolerance_bits)
        << " or later (" << tolerance_bits << " tolerance bits)\n";
    if (all_below_min_signal) {
        msg << "All values below min_signal: " << min_signal << std::endl;
    } else {
        msg << below_min_count << " value(s) below min_signal: " << min_signal << std::endl;
        msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << "tightest match   - mismatch occurred @ mantissa bit: " << matching_mantissa_bits(min_distance)
            << " or next bit (" << a[min_distance_index] << " vs " << b[min_distance_index] << " at ["
            << min_distance_index << "])\n";
        msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << "loosest match    - mismatch occurred @ mantissa bit: " << matching_mantissa_bits(max_distance)
            << " or next bit (" << a[max_distance_index] << " vs " << b[max_distance_index] << " at ["
            << max_distance_index << "])\n";
        msg << "median match     - mismatch occurred @ mantissa bit: " << matching_mantissa_bits(median_distance)
            << " or next bit\n";
    }

    ::testing::AssertionResult res = rc ? ::testing::AssertionSuccess() : ::testing::AssertionFailure();
    res << msg.str();
    return res;
}

::testing::AssertionResult all_close_f(const double* const a,
                                       const double* const b,
                                       size_t size,
                                       int tolerance_bits,
                                       double min_signal) {
    if (tolerance_bits < 0) {
        tolerance_bits = 0;
    }
    if (tolerance_bits >= DOUBLE_MANTISSA_BITS) {
        tolerance_bits = DOUBLE_MANTISSA_BITS - 1;
    }

    bool rc = true;
    std::stringstream msg;
    std::vector<uint64_t> distances = float_distances(a, b, size, min_signal);

    // e.g. for double with 52 bit mantissa, 2 bit accuracy, and hard-coded 11 bit exponent_bits
    // tolerance_bit_shift = 64 -           (1 +  11 + (53 -     1         ) - 2             )
    //                       double_length   sign exp   mantissa implicit 1    tolerance_bits
    uint64_t tolerance_bit_shift = 64 - (1 + 11 + (DOUBLE_MANTISSA_BITS - 1) - tolerance_bits);
    uint64_t tolerance = static_cast<uint64_t>(1U) << tolerance_bit_shift;
    uint64_t max_distance = 0;
    uint64_t min_distance = DOUBLE_BELOW_MIN_SIGNAL;
    size_t max_distance_index = 0;
    size_t min_distance_index = 0;
    size_t diff_count = 0;
    size_t below_min_count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (distances[i] == DOUBLE_BELOW_MIN_SIGNAL) {
            // Special value that indicates both values were below min_signal
            below_min_count++;
            continue;
        }

        if (distances[i] > max_distance) {
            max_distance = distances[i];
            max_distance_index = i;
        }
        if (distances[i] < min_distance) {
            min_distance = distances[i];
            min_distance_index = i;
        }
        bool is_close_f = distances[i] <= tolerance;
        if (!is_close_f) {
            if (diff_count < 5) {
                msg << a[i] << " is not close to " << b[i] << " at index " << i << std::endl;
            }

            rc = false;
            diff_count++;
        }
    }
    if (!rc) {
        msg << "diff count: " << diff_count << " out of " << size << std::endl;
    }
    // Find median value via partial sorting
    size_t middle = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + middle, distances.end());
    uint64_t median_distance = distances[middle];
    if (distances.size() % 2 == 0) {
        uint64_t median_distance2 = *max_element(distances.begin(), distances.begin() + middle);
        uint64_t remainder1 = median_distance % 2;
        uint64_t remainder2 = median_distance2 % 2;
        median_distance = (median_distance / 2) + (median_distance2 / 2) + ((remainder1 + remainder2) / 2);
    }

    bool all_below_min_signal = below_min_count == distances.size();
    if (rc) {
        // Short unobtrusive message when passing
        std::cout << "[   INFO   ] Verifying match of >= " << (DOUBLE_MANTISSA_BITS - tolerance_bits)
                  << " mantissa bits (" << DOUBLE_MANTISSA_BITS << " bits precision - " << tolerance_bits
                  << " tolerance). ";
        if (all_below_min_signal) {
            std::cout << "All values below min_signal: " << min_signal << std::endl;
        } else {
            std::cout << below_min_count << " value(s) below min_signal: " << min_signal << " Loosest match found is "
                      << matching_mantissa_bits(max_distance) << " mantissa bits.\n";
        }
    }

    msg << "passing criteria - mismatch allowed  @ mantissa bit: " << (DOUBLE_MANTISSA_BITS - tolerance_bits)
        << " or later (" << tolerance_bits << " tolerance bits)\n";
    if (all_below_min_signal) {
        msg << "All values below min_signal: " << min_signal << std::endl;
    } else {
        msg << below_min_count << " value(s) below min_signal: " << min_signal << std::endl;
        msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << "tightest match   - mismatch occurred @ mantissa bit: " << matching_mantissa_bits(min_distance)
            << " or next bit (" << a[min_distance_index] << " vs " << b[min_distance_index] << " at ["
            << min_distance_index << "])\n";
        msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << "loosest match    - mismatch occurred @ mantissa bit: " << matching_mantissa_bits(max_distance)
            << " or next bit (" << a[max_distance_index] << " vs " << b[max_distance_index] << " at ["
            << max_distance_index << "])\n";
        msg << "median match     - mismatch occurred @ mantissa bit: " << matching_mantissa_bits(median_distance)
            << " or next bit\n";
    }

    ::testing::AssertionResult res = rc ? ::testing::AssertionSuccess() : ::testing::AssertionFailure();
    res << msg.str();
    return res;
}

::testing::AssertionResult all_close_f(const std::vector<float>& a,
                                       const std::vector<float>& b,
                                       int tolerance_bits,
                                       float min_signal) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << "a.size() != b.size() for all_close_f comparison.";
    }
    if (a.size() == 0) {
        return ::testing::AssertionSuccess() << "No elements to compare";
    }

    return all_close_f(a.data(), b.data(), a.size(), tolerance_bits, min_signal);
}

::testing::AssertionResult all_close_f(const std::vector<double>& a,
                                       const std::vector<double>& b,
                                       int tolerance_bits,
                                       double min_signal) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << "a.size() != b.size() for all_close_f comparison.";
    }
    if (a.size() == 0) {
        return ::testing::AssertionSuccess() << "No elements to compare";
    }

    return all_close_f(a.data(), b.data(), a.size(), tolerance_bits, min_signal);
}

template <typename T>
::testing::AssertionResult all_close_f(const ov::Tensor& a, const ov::Tensor& b, int tolerance_bits, float min_signal) {
    if (a.get_size() != b.get_size()) {
        return ::testing::AssertionFailure() << "a.size() != b.size() for all_close_f comparison.";
    }
    if (a.get_size() == 0) {
        return ::testing::AssertionSuccess() << "No elements to compare";
    }

    return all_close_f(static_cast<const T*>(a.data()),
                       static_cast<const T*>(b.data()),
                       a.get_size(),
                       tolerance_bits,
                       min_signal);
}

::testing::AssertionResult all_close_f(const ov::Tensor& a, const ov::Tensor& b, int tolerance_bits, float min_signal) {
    if (a.get_element_type() != b.get_element_type()) {
        return ::testing::AssertionFailure() << "Cannot compare tensors with different element types";
    }

#define all_close_f_ov_type(type) \
    case ov::element::type:       \
        return all_close_f<ov::element_type_traits<ov::element::type>::value_type>(a, b, tolerance_bits, min_signal);

    switch (a.get_element_type()) {
        // all_close_f_ov_type(u8)
        // all_close_f_ov_type(u16)
        // all_close_f_ov_type(u32)
        // all_close_f_ov_type(u64)
        // all_close_f_ov_type(i8)
        // all_close_f_ov_type(i16)
        // all_close_f_ov_type(i32)
        // all_close_f_ov_type(i64)
        // all_close_f_ov_type(bf16)
        // all_close_f_ov_type(f16)
        all_close_f_ov_type(f32) all_close_f_ov_type(f64) default
            : return ::testing::AssertionFailure()
              << "Cannot compare tensors with unsupported element type: "
              << a.get_element_type();
    }
}

::testing::AssertionResult all_close_f(const std::vector<ov::Tensor>& as,
                                       const std::vector<ov::Tensor>& bs,
                                       int tolerance_bits,
                                       float min_signal) {
    if (as.size() != bs.size()) {
        return ::testing::AssertionFailure() << "Cannot compare tensors with different sizes";
    }
    for (size_t i = 0; i < as.size(); ++i) {
        auto ar = all_close_f(as[i], bs[i], tolerance_bits, min_signal);
        if (!ar) {
            return ar;
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace utils
}  // namespace test
}  // namespace ov
