// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"

#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_iterator.hpp"
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
    auto size = shape_size(shape);

#define CASE(X)                                                  \
    case X:                                                      \
        fill_data_random(tensor.data<fundamental_type_for<X>>(), \
                         size,                                   \
                         inGenData.range,                        \
                         inGenData.start_from,                   \
                         inGenData.resolution,                   \
                         inGenData.seed);                        \
        break;

#define CASE_CONVERT(X)                                          \
    case X: {                                                    \
        auto input = std::vector<fundamental_type_for<X>>(size); \
        fill_data_random(input.data(),                           \
                         size,                                   \
                         inGenData.range,                        \
                         inGenData.start_from,                   \
                         inGenData.resolution,                   \
                         inGenData.seed);                        \
        auto iter = element::iterator<X>(tensor.data());         \
        std::copy(input.begin(), input.end(), iter);             \
        break;                                                   \
    }

    switch (element_type) {
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
        CASE_CONVERT(ov::element::u6)
        CASE_CONVERT(ov::element::u4)
        CASE_CONVERT(ov::element::u3)
        CASE_CONVERT(ov::element::u2)
        CASE_CONVERT(ov::element::u1)
        CASE_CONVERT(ov::element::i4)
        CASE_CONVERT(ov::element::nf4)
        CASE_CONVERT(ov::element::f8e4m3)
        CASE_CONVERT(ov::element::f8e5m2)
        CASE_CONVERT(ov::element::f8e8m0)
        CASE_CONVERT(ov::element::f4e2m1)
    case ov::element::boolean:
        fill_data_boolean(static_cast<fundamental_type_for<ov::element::boolean>*>(tensor.data()),
                          size,
                          inGenData.seed);
        break;
    case ov::element::Type_t::string:
        fill_random_string(static_cast<std::string*>(tensor.data()),
                           size,
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
        CASE(ov::element::Type_t::f8e8m0)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
    case ov::element::Type_t::nf4:
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

namespace tensor_comparation {
constexpr double eps = std::numeric_limits<double>::epsilon();

inline bool less(const double a, const double b) {
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    } else if (std::isinf(b) && std::isinf(b)) {
        return true;
    } else if (std::isinf(b) && b > 0) {
        // b is greater than any number or eq the +Inf
        return true;
    } else if (std::isinf(a) && a > 0) {
        return false;
    }
    return std::fabs(a - b) > eps && a < b;
}

inline bool equal(const double a, const double b) {
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    } else if (std::isinf(b) && std::isinf(b)) {
        return true;
    } else if (std::isinf(b) && b > 0) {
        // b is greater than any number or eq the +Inf
        return true;
    } else if (std::isinf(a) && a > 0) {
        return false;
    }
    return std::fabs(b - a) <= eps;
}

inline bool less_or_equal(double a, double b) {
    return less(a, b) || equal(a, b);
}

template <typename T1, typename T2>
inline bool is_value_suitable_for_comparation(const double value1, const double value2) {
    bool res = true;
    auto max_val1 = std::numeric_limits<T1>::max();
    auto min_val1 = std::numeric_limits<T1>::lowest();
    auto max_val2 = std::numeric_limits<T2>::max();
    auto min_val2 = std::numeric_limits<T2>::lowest();
    if (std::isnan(value1) && std::isnan(value2)) {
        res = false;
    } else if ((std::isinf(value1) || value1 >= max_val1) && (std::isinf(value2) || value2 >= max_val2)) {
        res = false;
    } else if ((std::isinf(value1) || value1 <= min_val1) && std::isinf(value2) || value2 <= min_val2) {
        res = false;
    }
    return res;
}

class Error {
protected:
    struct IncorrectValue {
        size_t coordinate;
        double actual_value, expected_value, threshold;

        IncorrectValue(double in_actual_value, double in_expected_value, double in_threshold, size_t in_coordinate)
            : actual_value(in_actual_value),
              expected_value(in_expected_value),
              threshold(in_threshold),
              coordinate(in_coordinate) {}
    };

    std::vector<IncorrectValue> incorrect_values_abs;
    double abs_threshold, rel_threshold, mvn_threshold, topk_threshold, mvn_results, topk_results;
    size_t tensor_size;

    void emplace_back(double in_actual_value, double in_expected_value, double in_threshold, size_t in_coordinate) {
        incorrect_values_abs.push_back(IncorrectValue(in_actual_value, in_expected_value, in_threshold, in_coordinate));
    }

public:
    Error(const double in_abs_threshold,
          const double in_rel_threshold,
          const double in_topk_threshold,
          const double in_mvn_threshold,
          int in_tensor_size)
        : abs_threshold(in_abs_threshold),
          rel_threshold(in_rel_threshold),
          mvn_threshold(in_mvn_threshold),
          topk_threshold(in_topk_threshold),
          tensor_size(in_tensor_size),
          mvn_results(0.f),
          topk_results(0.f) {}

    bool update(double actual, double expected, size_t coordinate) {
        const auto diff = std::fabs(expected - actual);
        const auto threshold = calculate_threshold(abs_threshold, rel_threshold, expected);
        mvn_results += equal(threshold, 0.f) ? diff : (diff / threshold);
        if (less_or_equal(diff, threshold)) {
            return true;
        }
        emplace_back(actual, expected, threshold, coordinate);
        return false;
    }

    void check_results() {
        topk_results = static_cast<double>(incorrect_values_abs.size()) / (tensor_size ? tensor_size : 1);
        mvn_results /= tensor_size ? tensor_size : 1;
        if (!incorrect_values_abs.empty() && equal(1.f, topk_threshold) ||
            incorrect_values_abs.size() > static_cast<int>(std::floor(topk_threshold * tensor_size))) {
            std::string msg = "[ COMPARATION ] COMPARATION IS FAILED! incorrect elem counter: ";
            msg += std::to_string(incorrect_values_abs.size());
            msg += " among ";
            msg += std::to_string(tensor_size);
            msg += " shapes.";
            for (auto val : incorrect_values_abs) {
                std::cout << "\nExpected: " << val.expected_value << " Actual: " << val.actual_value
                          << " Coordinate: " << val.coordinate
                          << " Diff: " << std::fabs(val.expected_value - val.actual_value)
                          << " calculated_abs_threshold: " << val.threshold << " abs_threshold: " << abs_threshold
                          << " rel_threshold: " << rel_threshold << "\n";
#ifdef NDEBUG
                break;
#endif
            }
            throw std::runtime_error(msg);
        } else if (!less_or_equal(mvn_results, mvn_threshold)) {
            std::string msg = "[ COMPARATION ] COMPARATION IS FAILED due to MVN THRESHOLD: ";
            msg += std::to_string(mvn_threshold);
            msg += ". Actual MVN value is ";
            msg += std::to_string(mvn_results);
            throw std::runtime_error(msg);
        }
    }
};

double calculate_threshold(const double abs_threshold, const double rel_threshold, const double ref_value) {
    return abs_threshold + rel_threshold * std::fabs(ref_value);
}

double calculate_default_abs_threshold(const ov::element::Type& expected_type,
                                       const ov::element::Type& actual_type,
                                       const ov::element::Type& inference_precision) {
    const std::vector<ov::element::Type> element_types{expected_type, actual_type, inference_precision};
    std::vector<double> values;
#define CASE(X)                                                                               \
    case X:                                                                                   \
        values.push_back(std::numeric_limits<element_type_traits<X>::value_type>::epsilon()); \
        break;

    for (const auto& elem_type : element_types) {
        switch (elem_type) {
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
            CASE(ov::element::Type_t::nf4)
            CASE(ov::element::Type_t::f8e4m3)
            CASE(ov::element::Type_t::f8e5m2)
        default:
            values.push_back(0.f);
            break;
        }
    }
#undef CASE

    double threshold = *std::max_element(values.begin(), values.end());
    return threshold;
}

double calculate_default_rel_threshold(const ov::element::Type& expected_type,
                                       const ov::element::Type& actual_type,
                                       const ov::element::Type& inference_precision) {
    std::vector<double> values{get_eps_by_ov_type(expected_type),
                               get_eps_by_ov_type(actual_type),
                               get_eps_by_ov_type(inference_precision)};
    double threshold = *std::max_element(values.begin(), values.end());
    return threshold;
}

}  // namespace tensor_comparation

template <typename ExpectedT, typename ActualT>
void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             const ov::element::Type& inference_precision,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    // check shapes
    auto expected_shape = expected.get_shape();
    auto actual_shape = actual.get_shape();
    if (expected_shape != actual_shape) {
        std::ostringstream out_stream;
        out_stream << "Expected and actual shape are different: " << expected_shape << " " << actual_shape;
        throw std::runtime_error(out_stream.str());
    } else if (shape_size(actual_shape) == 0) {
        return;
    }

    // Set default values in case threshold values are incorrect
    const auto expected_type = expected.get_element_type();
    const auto actual_type = actual.get_element_type();
    if (abs_threshold < 0) {
        abs_threshold =
            tensor_comparation::calculate_default_abs_threshold(expected_type, actual_type, inference_precision);
    }
    if (rel_threshold < 0) {
        rel_threshold =
            tensor_comparation::calculate_default_rel_threshold(expected_type, actual_type, inference_precision);
    }
    if (topk_threshold < 0.f || topk_threshold > 1.f) {
        topk_threshold = 1.f;
        std::cout << "[ WARNING ] Incorrect value: " << topk_threshold
                  << " for Topk_threshold. It should be [0.f, 1.f]. Reset default value is 1.f" << std::endl;
    }
    if (mvn_threshold < 0.f || mvn_threshold > 1.f) {
        mvn_threshold = 1.f;
        std::cout << "[ WARNING ] Incorrect value: " << mvn_threshold
                  << " for MVN_threshold. It should be [0.f, 1.f]. Reset default value is 1.f" << std::endl;
    }

    // error is a place with whole data related to incorrect element in tensor
    size_t shape_size_cnt = shape_size(expected_shape);
    tensor_comparation::Error error(abs_threshold, rel_threshold, topk_threshold, mvn_threshold, shape_size_cnt);
    const auto expected_data = expected.data<ExpectedT>();
    const auto actual_data = actual.data<ActualT>();
    for (size_t i = 0; i < shape_size_cnt; ++i) {
        double expected_value = expected_data[i];
        double actual_value = actual_data[i];
        if (!tensor_comparation::is_value_suitable_for_comparation<ExpectedT, ActualT>(expected_value, actual_value)) {
            continue;
        }

        bool status = error.update(actual_value, expected_value, i);
#ifdef NDEBUG
        if (!status && tensor_comparation::equal(topk_threshold, 1.f)) {
            break;
        }
#endif
    }
    error.check_results();
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
             const ov::element::Type& inference_precision,
             const double abs_threshold,
             const double rel_threshold,
             const double topk_threshold,
             const double mvn_threshold) {
#define CASE0(X, Y)                                                                                          \
    case Y:                                                                                                  \
        compare<element_type_traits<X>::value_type, element_type_traits<Y>::value_type>(expected,            \
                                                                                        actual,              \
                                                                                        inference_precision, \
                                                                                        abs_threshold,       \
                                                                                        rel_threshold,       \
                                                                                        topk_threshold,      \
                                                                                        mvn_threshold);      \
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
            CASE0(X, ov::element::Type_t::f8e4m3)        \
            CASE0(X, ov::element::Type_t::f8e5m2)        \
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
        CASE(ov::element::Type_t::f8e4m3)
        CASE(ov::element::Type_t::f8e5m2)
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
