// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"

#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "precomp.hpp"

namespace ov {
namespace test {
namespace utils {
ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const uint32_t range,
                                  const double_t start_from,
                                  const int32_t resolution,
                                  const int seed) {
    auto tensor = ov::Tensor{element_type, shape};
#define CASE(X)                                                          \
    case X:                                                              \
        fill_tensor_random(tensor, range, start_from, resolution, seed); \
        break;
    switch (element_type) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
    case ov::element::Type_t::nf4:
        fill_data_random(static_cast<uint8_t*>(tensor.data()),
                         tensor.get_byte_size(),
                         range,
                         start_from,
                         resolution,
                         seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

ov::Tensor create_and_fill_tensor_unique_sequence(const ov::element::Type element_type,
                                                  const ov::Shape& shape,
                                                  const int32_t start_from,
                                                  const int32_t resolution,
                                                  const int seed) {
    auto tensor = ov::Tensor{element_type, shape};
    auto range = shape_size(shape) * 2;
#define CASE(X)                                                                        \
    case X:                                                                            \
        fill_random_unique_sequence(tensor.data<element_type_traits<X>::value_type>(), \
                                    shape_size(shape),                                 \
                                    range,                                             \
                                    start_from,                                        \
                                    resolution,                                        \
                                    seed);                                             \
        break;

    switch (element_type) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
        fill_random_unique_sequence(static_cast<uint8_t*>(tensor.data()),
                                    tensor.get_byte_size(),
                                    range,
                                    start_from,
                                    resolution,
                                    seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

ov::runtime::Tensor create_and_fill_tensor_normal_distribution(const ov::element::Type element_type,
                                                               const ov::Shape& shape,
                                                               const float mean,
                                                               const float stddev,
                                                               const int seed) {
    auto tensor = ov::runtime::Tensor{element_type, shape};
#define CASE(X)                                                                              \
    case X:                                                                                  \
        fill_data_ptr_normal_random_float(tensor.data<element_type_traits<X>::value_type>(), \
                                          shape_size(shape),                                 \
                                          mean,                                              \
                                          stddev,                                            \
                                          seed);                                             \
        break;
    switch (element_type) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
        fill_data_ptr_normal_random_float(static_cast<uint8_t*>(tensor.data()),
                                          tensor.get_byte_size(),
                                          mean,
                                          stddev,
                                          seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

ov::runtime::Tensor create_and_fill_tensor_consistently(const ov::element::Type element_type,
                                                        const ov::Shape& shape,
                                                        const uint32_t range,
                                                        const int32_t start_from,
                                                        const int32_t resolution) {
    auto tensor = ov::runtime::Tensor{element_type, shape};
#define CASE(X)                                                                       \
    case X:                                                                           \
        fill_data_ptr_consistently(tensor.data<element_type_traits<X>::value_type>(), \
                                   tensor.get_size(),                                 \
                                   range,                                             \
                                   start_from,                                        \
                                   resolution);                                       \
        break;
    switch (element_type) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
        fill_data_ptr_consistently(static_cast<uint8_t*>(tensor.data()),
                                   tensor.get_byte_size(),
                                   range,
                                   start_from,
                                   resolution);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

constexpr double eps = std::numeric_limits<double>::epsilon();

inline double less(double a, double b) {
    return (b - a) > (std::fmax(std::fabs(a), std::fabs(b)) * eps);
}

inline double less_or_equal(double a, double b) {
    return (b - a) >= (std::fmax(std::fabs(a), std::fabs(b)) * eps);
}

struct Error {
    double max = 0.;
    double mean = 0.;
    size_t max_coordinate = 0;
    size_t count = 0;
    double threshold;

    Error(double _threshold) : threshold(_threshold) {}

    void update(double val, size_t i) {
        if (less(max, val)) {
            max = val;
            max_coordinate = i;
        }
        mean += val;
        count += less(threshold, val);
    }
};

inline double calculate_median(std::vector<double>& abs_values) {
    double abs_median = 0.;
    auto expected_shape = abs_values.size();
    if (expected_shape % 2) {
        std::nth_element(abs_values.begin(), abs_values.begin() + expected_shape / 2, abs_values.end());
        abs_median = abs_values[expected_shape / 2];
    } else {
        std::nth_element(abs_values.begin(), abs_values.begin() + expected_shape / 2, abs_values.end());
        std::nth_element(abs_values.begin(), abs_values.begin() + (expected_shape - 1) / 2, abs_values.end());
        abs_median = (abs_values[(expected_shape - 1) / 2] + abs_values[expected_shape / 2]) / 2.0;
    }
    return abs_median;
}

template <typename ExpectedT, typename ActualT>
void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             const double abs_threshold_ = std::numeric_limits<double>::max(),
             const double rel_threshold_ = std::numeric_limits<double>::max()) {
    auto expected_shape = expected.get_shape();
    auto actual_shape = actual.get_shape();
    if (expected_shape != actual_shape) {
        std::ostringstream out_stream;
        out_stream << "Expected and actual shape are different: " << expected_shape << " " << actual_shape;
        throw std::runtime_error(out_stream.str());
    }
    if (shape_size(actual_shape) == 0) {
        return;
    }

    auto expected_data = expected.data<ExpectedT>();
    auto actual_data = actual.data<ActualT>();
    double abs_threshold = abs_threshold_;
    double rel_threshold = rel_threshold_;
    size_t shape_size_cnt = shape_size(expected_shape);
    if (abs_threshold == std::numeric_limits<double>::max() && rel_threshold == std::numeric_limits<double>::max()) {
        if (sizeof(ExpectedT) == 1 || sizeof(ActualT) == 1) {
            abs_threshold = 1.;
        } else {
            std::vector<double> abs_values(shape_size_cnt);
            for (size_t i = 0; i < shape_size_cnt; i++) {
                abs_values[i] = std::fabs(static_cast<double>(expected_data[i]));
            }
            auto abs_median = calculate_median(abs_values);
            abs_threshold = abs_median * 0.05 < 1e-5 ? 1e-5 : 0.05 * abs_median;
            if (std::is_integral<ExpectedT>::value) {
                abs_threshold = std::ceil(abs_threshold);
            }
        }
    }
    if (!std::isnan(abs_threshold)) {
        std::cout << "[ COMPARATION ] rel_threshold: " << rel_threshold << std::endl;
    }
    if (!std::isnan(rel_threshold)) {
        std::cout << "[ COMPARATION ] abs_threshold: " << abs_threshold << std::endl;
    }

    Error abs_error(abs_threshold), rel_error(rel_threshold);
    for (size_t i = 0; i < shape_size_cnt; ++i) {
        double expected_value = expected_data[i];
        double actual_value = actual_data[i];
        if (std::isnan(expected_value)) {
            std::ostringstream out_stream;
            out_stream << "Expected value is NAN on coordinate: " << i;
            throw std::runtime_error(out_stream.str());
        }
        if (std::isnan(actual_value)) {
            std::ostringstream out_stream;
            out_stream << "Actual value is NAN on coordinate: " << i;
            throw std::runtime_error(out_stream.str());
        }
        double abs = std::fabs(expected_value - actual_value);
        double rel = expected_value ? (abs / std::fabs(expected_value)) : abs;
        abs_error.update(abs, i);
        rel_error.update(rel, i);
    }
    abs_error.mean /= shape_size_cnt;
    rel_error.mean /= shape_size_cnt;

    if (!(less_or_equal(abs_error.max, abs_threshold) && less_or_equal(rel_error.max, rel_threshold))) {
        std::ostringstream out_stream;
        out_stream << "abs_max < abs_threshold && rel_max < rel_threshold"
                   << "\n\t abs_max: " << abs_error.max << "\n\t\t coordinate " << abs_error.max_coordinate
                   << "; abs errors count " << abs_error.count << "; abs mean " << abs_error.mean << "; abs threshold "
                   << abs_threshold << "\n\t rel_max: " << rel_error.max << "\n\t\t coordinate "
                   << rel_error.max_coordinate << "; rel errors count " << rel_error.count << "; rel mean "
                   << rel_error.mean << "; rel threshold " << rel_threshold;
        throw std::runtime_error(out_stream.str());
    }
}

void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             const double abs_threshold,
             const double rel_threshold) {
#define CASE0(X, Y)                                                                                     \
    case Y:                                                                                             \
        compare<element_type_traits<X>::value_type, element_type_traits<Y>::value_type>(expected,       \
                                                                                        actual,         \
                                                                                        abs_threshold,  \
                                                                                        rel_threshold); \
        break;

#define CASE(X)                                          \
    case X:                                              \
        switch (actual.get_element_type()) {             \
            CASE0(X, ov::element::Type_t::boolean)       \
            CASE0(X, ov::element::Type_t::bf16)          \
            CASE0(X, ov::element::Type_t::f16)           \
            CASE0(X, ov::element::Type_t::f32)           \
            CASE0(X, ov::element::Type_t::f64)           \
            CASE0(X, ov::element::Type_t::i4)            \
            CASE0(X, ov::element::Type_t::i8)            \
            CASE0(X, ov::element::Type_t::i16)           \
            CASE0(X, ov::element::Type_t::i32)           \
            CASE0(X, ov::element::Type_t::i64)           \
            CASE0(X, ov::element::Type_t::u1)            \
            CASE0(X, ov::element::Type_t::u4)            \
            CASE0(X, ov::element::Type_t::u8)            \
            CASE0(X, ov::element::Type_t::u16)           \
            CASE0(X, ov::element::Type_t::u32)           \
            CASE0(X, ov::element::Type_t::u64)           \
        default:                                         \
            OPENVINO_THROW("Unsupported element type: ", \
                           "expected ",                  \
                           expected.get_element_type(),  \
                           ", actual ",                  \
                           actual.get_element_type());   \
        }                                                \
        break;

    switch (expected.get_element_type()) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
        CASE(ov::element::Type_t::i4)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u1)
        CASE(ov::element::Type_t::u4)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
    default:
        OPENVINO_THROW("Unsupported element type: ", expected.get_element_type());
    }
#undef CASE0
#undef CASE
}
}  // namespace utils
}  // namespace test
}  // namespace ov
