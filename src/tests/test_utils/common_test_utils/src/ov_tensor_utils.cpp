// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <math.h>

#include "common_test_utils/ov_tensor_utils.hpp"

#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"
#include "precomp.hpp"

namespace ov {
namespace test {
namespace utils {

double ConstRanges::max = std::numeric_limits<double>::min();
double ConstRanges::min = std::numeric_limits<double>::max();
bool ConstRanges::is_defined = false;

ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const InputGenerateData& inGenData) {
    auto tensor = ov::Tensor(element_type, shape);

#define CASE(X)                                                  \
    case X:                                                      \
        fill_data_random(tensor.data<fundamental_type_for<X>>(), \
                         shape_size(shape),                      \
                         inGenData.range,                        \
                         inGenData.start_from,                   \
                         inGenData.resolution,                   \
                         inGenData.seed);                        \
        break;

    switch (element_type) {
        CASE(ov::element::boolean)
        CASE(ov::element::i8)
        CASE(ov::element::i16)
        CASE(ov::element::i32)
        CASE(ov::element::i64)
        CASE(ov::element::u8)
        CASE(ov::element::u16)
        CASE(ov::element::u32)
        CASE(ov::element::u64)
        CASE(ov::element::bf16)
        CASE(ov::element::f16)
        CASE(ov::element::f32)
        CASE(ov::element::f64)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
    case ov::element::Type_t::nf4:
        fill_data_random(static_cast<uint8_t*>(tensor.data()),
                         tensor.get_byte_size(),
                         inGenData.range,
                         inGenData.start_from,
                         inGenData.resolution,
                         inGenData.seed);
        break;
    case ov::element::Type_t::string:
        fill_random_string(static_cast<std::string*>(tensor.data()),
                           tensor.get_size(),
                           inGenData.range,
                           inGenData.start_from,
                           inGenData.seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

// Legacy impl for contrig repo
// todo: remove this after dependent repos clean up
ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const uint32_t range,
                                  const double_t start_from,
                                  const int32_t resolution,
                                  const int seed) {
    return create_and_fill_tensor(element_type,
                                  shape,
                                  ov::test::utils::InputGenerateData(start_from, range, resolution, seed));
}

ov::Tensor create_and_fill_tensor_act_dft(const ov::element::Type element_type,
                                          const ov::Shape& shape,
                                          const uint32_t range,
                                          const double_t start_from,
                                          const int32_t resolution,
                                          const int seed) {
    auto tensor = ov::Tensor{element_type, shape};
#define CASE(X)                                                                     \
    case X:                                                                         \
        fill_data_random_act_dft(tensor.data<element_type_traits<X>::value_type>(), \
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
    case ov::element::Type_t::nf4:
        fill_data_random_act_dft(static_cast<uint8_t*>(tensor.data()),
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

ov::Tensor create_and_fill_tensor_normal_distribution(const ov::element::Type element_type,
                                                      const ov::Shape& shape,
                                                      const float mean,
                                                      const float stddev,
                                                      const int seed) {
    auto tensor = ov::Tensor{element_type, shape};
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

ov::Tensor create_and_fill_tensor_real_distribution(const ov::element::Type element_type,
                                                    const ov::Shape& shape,
                                                    const float min,
                                                    const float max,
                                                    const int seed) {
    auto tensor = ov::Tensor{element_type, shape};
#define CASE(X)                                                                            \
    case X:                                                                                \
        fill_data_ptr_real_random_float(tensor.data<element_type_traits<X>::value_type>(), \
                                        shape_size(shape),                                 \
                                        min,                                               \
                                        max,                                               \
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
        fill_data_ptr_real_random_float(static_cast<uint8_t*>(tensor.data()), tensor.get_byte_size(), min, max, seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

ov::Tensor create_and_fill_tensor_consistently(const ov::element::Type element_type,
                                               const ov::Shape& shape,
                                               const uint32_t range,
                                               const int32_t start_from,
                                               const int32_t resolution) {
    auto tensor = ov::Tensor{element_type, shape};
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
    return std::fabs(a - b) > eps && a < b;
}

inline double less_or_equal(double a, double b) {
    bool res = true;
    if (std::isnan(a) || std::isnan(b)) {
        res = false;
    } else if (std::isinf(b) && b > 0) {
        // b is greater than any number or eq the +Inf
        res = true;
    } else if (std::isinf(a) && a > 0) {
        res = false;
    } else {
        res = std::fabs(b - a) <= eps || a <= b;
    }
    return res;
}

template <typename T>
inline bool check_value(const T& value) {
    if (std::isnan(value)||
        std::isinf(value) ||
        value >= std::numeric_limits<T>::max() ||
        value <= std::numeric_limits<T>::min()) {
        return false;
    }
    return true;
}

class Error {
protected:
    struct IncorrectValue {
        size_t coordinate;
        double actual_value, expected_value, abs_threshold, rel_threshold; 

        IncorrectValue(
            double in_actual_value,
            double in_expected_value,
            double in_abs_threshold,
            double in_rel_threshold,
            size_t in_coordinate) :
            actual_value(in_actual_value),
            expected_value(in_expected_value),
            abs_threshold(in_abs_threshold),
            rel_threshold(in_rel_threshold),
            coordinate(in_coordinate) {}
    };

    std::vector<IncorrectValue> incorrect_values;
    double abs_threshold, rel_threshold;

public:
    Error(const double in_abs_threshold,
          const double in_rel_threshold) : 
          abs_threshold(in_abs_threshold),
          rel_threshold(in_rel_threshold) {}

    void update(double actual, double expected, size_t coordinate) {
        const auto diff = std::fabs(expected - actual);

        const auto calculated_abs_threshold = abs_threshold * expected;
        const auto calculated_rel_threshold = calculated_abs_threshold ? diff / calculated_abs_threshold : 1.;
        if (less_or_equal(diff, calculated_abs_threshold) &&
            less_or_equal(calculated_rel_threshold, rel_threshold)) {
            return;
        }
        incorrect_values.emplace_back(IncorrectValue(actual, expected, calculated_abs_threshold,
                                                     calculated_rel_threshold, coordinate));
    }

    void get_results() {
        if (!incorrect_values.empty()) {
            std::string msg = "[ COMPARATION ] COMPARATION IS FAILED! incorrect elem counter: ";
            msg += std::to_string(incorrect_values.size());
            msg += ". Please print `incorrect_values` to get detailed information!";
            throw std::runtime_error(msg);
        }
    }
};

template <typename ExpectedT, typename ActualT>
void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             double abs_threshold,
             const double rel_threshold) {
    auto expected_shape = expected.get_shape();
    auto actual_shape = actual.get_shape();
    if (expected_shape != actual_shape) {
        std::ostringstream out_stream;
        out_stream << "Expected and actual shape are different: " << expected_shape << " " << actual_shape;
        throw std::runtime_error(out_stream.str());
    } else if (shape_size(actual_shape) == 0) {
        return;
    }

    const auto default_abs_threshold = get_eps_by_ov_type(expected.get_element_type());
    if (abs_threshold == std::numeric_limits<double>::max()) {
        abs_threshold = default_abs_threshold;
    } else if (abs_threshold < default_abs_threshold) {
        abs_threshold = default_abs_threshold;
    }

    size_t shape_size_cnt = shape_size(expected_shape);
    Error error(abs_threshold, rel_threshold);
    const auto expected_data = expected.data<ExpectedT>();
    const auto actual_data = actual.data<ActualT>();
    for (size_t i = 0; i < shape_size_cnt; ++i) {
        double expected_value = expected_data[i];
        double actual_value = actual_data[i];
        if (!check_value(expected_value) && !check_value(actual_value)) {
            continue;
        }
        error.update(expected_value, actual_value, i);
    }
    error.get_results();
}

void compare_str(const ov::Tensor& expected, const ov::Tensor& actual) {
    ASSERT_EQ(expected.get_element_type(), ov::element::string);
    ASSERT_EQ(actual.get_element_type(), ov::element::string);
    EXPECT_EQ(expected.get_shape(), actual.get_shape());

    const auto expected_const = ov::op::v0::Constant(expected);
    const auto result_const = ov::op::v0::Constant(actual);
    EXPECT_EQ(expected_const.get_value_strings(), result_const.get_value_strings());
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
    case ov::element::Type_t::string:
        compare_str(expected, actual);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", expected.get_element_type());
    }
#undef CASE0
#undef CASE
}
}  // namespace utils
}  // namespace test
}  // namespace ov
