// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"

#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>
#include <string>
#include <utility>

namespace cldnn {

namespace err_details {
void cldnn_print_error_message(const std::string& file,
                               int line,
                               const std::string& instance_id,
                               std::stringstream& msg,
                               const std::string& add_msg = "");
}

template <typename N1, typename N2>
inline void error_on_not_equal(const std::string& file,
                               int line,
                               const std::string& instance_id,
                               const std::string& number_id,
                               N1 number,
                               const std::string& compare_to_id,
                               N2 number_to_compare_to,
                               const std::string& additional_message = "") {
    if (number != static_cast<decltype(number)>(number_to_compare_to)) {
        std::stringstream error_msg;
        error_msg << number_id << "(=" << number << ") is not equal to: " << compare_to_id
                  << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_NOT_EQUAL(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) \
    error_on_not_equal(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template <typename N1, typename N2>
inline void error_on_equal(const std::string& file,
                           int line,
                           const std::string& instance_id,
                           const std::string& number_id,
                           N1 number,
                           const std::string& compare_to_id,
                           N2 number_to_compare_to,
                           const std::string& additional_message = "") {
    if (number == static_cast<decltype(number)>(number_to_compare_to)) {
        std::stringstream error_msg;
        error_msg << number_id << "(=" << number << ") is equal to: " << compare_to_id << "(=" << number_to_compare_to
                  << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_EQUAL(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) \
    error_on_equal(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template <typename N1, typename N2>
inline void error_on_greater_than(const std::string& file,
                                  int line,
                                  const std::string& instance_id,
                                  const std::string& number_id,
                                  N1 number,
                                  const std::string& compare_to_id,
                                  N2 number_to_compare_to,
                                  const std::string& additional_message = "") {
    if (number > static_cast<decltype(number)>(number_to_compare_to)) {
        std::stringstream error_msg;
        error_msg << number_id << "(=" << number << ") is greater than: " << compare_to_id
                  << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_GREATER_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) \
    error_on_greater_than(__FILE__,                                                                            \
                          __LINE__,                                                                            \
                          instance_id,                                                                         \
                          number_id,                                                                           \
                          number,                                                                              \
                          compare_to_id,                                                                       \
                          number_to_compare_to,                                                                \
                          add_msg)

template <typename N1, typename N2>
inline void error_on_less_than(const std::string& file,
                               int line,
                               const std::string& instance_id,
                               const std::string& number_id,
                               N1 number,
                               const std::string& compare_to_id,
                               N2 number_to_compare_to,
                               const std::string& additional_message = "") {
    if (number < static_cast<decltype(number)>(number_to_compare_to)) {
        std::stringstream error_msg;
        error_msg << number_id << "(=" << number << ") is less than: " << compare_to_id << "(=" << number_to_compare_to
                  << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_LESS_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) \
    error_on_less_than(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template <typename N1, typename N2>
inline void error_on_less_or_equal_than(const std::string& file,
                                        int line,
                                        const std::string& instance_id,
                                        const std::string& number_id,
                                        N1 number,
                                        const std::string& compare_to_id,
                                        N2 number_to_compare_to,
                                        const std::string& additional_message = "") {
    if (number <= static_cast<decltype(number)>(number_to_compare_to)) {
        std::stringstream error_msg;
        error_msg << number_id << "(=" << number << ") is less or equal than: " << compare_to_id
                  << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_LESS_OR_EQUAL_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) \
    error_on_less_or_equal_than(__FILE__,                                                                            \
                                __LINE__,                                                                            \
                                instance_id,                                                                         \
                                number_id,                                                                           \
                                number,                                                                              \
                                compare_to_id,                                                                       \
                                number_to_compare_to,                                                                \
                                add_msg)

template <typename N1, typename N2>
inline void error_on_greater_or_equal_than(const std::string& file,
                                           int line,
                                           const std::string& instance_id,
                                           const std::string& number_id,
                                           N1 number,
                                           const std::string& compare_to_id,
                                           N2 number_to_compare_to,
                                           const std::string& additional_message = "") {
    if (number >= static_cast<decltype(number)>(number_to_compare_to)) {
        std::stringstream error_msg;
        error_msg << number_id << "(=" << number << ") is greater or equal than: " << compare_to_id
                  << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_GREATER_OR_EQUAL_THAN(instance_id,          \
                                          number_id,            \
                                          number,               \
                                          compare_to_id,        \
                                          number_to_compare_to, \
                                          add_msg)              \
    error_on_greater_or_equal_than(__FILE__,                    \
                                   __LINE__,                    \
                                   instance_id,                 \
                                   number_id,                   \
                                   number,                      \
                                   compare_to_id,               \
                                   number_to_compare_to,        \
                                   add_msg)

template <typename ptr>
inline void error_on_nullptr(const std::string& file,
                             int line,
                             const std::string& instance_id,
                             const std::string& condition_id,
                             ptr condition,
                             const std::string& additional_message = "") {
    if (condition == nullptr) {
        std::stringstream error_msg;
        error_msg << condition_id << " should not be null" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_NULLPTR(instance_id, condition_id, condition, add_msg) \
    error_on_nullptr(__FILE__, __LINE__, instance_id, condition_id, condition, add_msg)

template <typename M = format, typename... Ms>
inline void error_on_not_proper_enum_values(const std::string& file,
                                            int line,
                                            const std::string& instance_id,
                                            const std::string& mode_id,
                                            M mode,
                                            const std::string& modes_id,
                                            Ms... modes_to_compare_to) {
    auto enum_value_string = [](const M& mode) -> std::string {
        if (std::is_same<M, format::type>::value) {
            return format::traits(mode).order;
        }
        return "error during error parsing";
    };
    const std::array<const M, sizeof...(Ms)> modes{std::forward<Ms>(modes_to_compare_to)...};
    if (std::all_of(modes.begin(), modes.end(), [&](const M& m) -> int { return mode != m; })) {
        std::stringstream error_msg;
        error_msg << mode_id << "( " << enum_value_string(mode) << " ) is incompatible with " << modes_id
                  << ". Should be one of: ";
        for (const auto& ms : modes) {
            error_msg << enum_value_string(ms) << ", ";
        }
        error_msg << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg);
    }
}
#define CLDNN_ERROR_NOT_PROPER_FORMAT(instance_id, format_id, formatt, formats_ids, ...) \
    error_on_not_proper_enum_values(__FILE__, __LINE__, instance_id, format_id, formatt, formats_ids, __VA_ARGS__)

void error_message(const std::string& file, int line, const std::string& instance_id, const std::string& message);
#define CLDNN_ERROR_MESSAGE(instance_id, message) error_message(__FILE__, __LINE__, instance_id, message)

void error_on_not_supported_fp16(const std::string& file,
                                 int line,
                                 const std::string& instance_id,
                                 uint8_t supp_fp16,
                                 bool fp16_used);
#define CLDNN_ERROR_NOT_SUPPORTED_FP16(instance_id, gpu_supp_fp16, fp16_used) \
    error_on_not_supported_fp16(__FILE__, __LINE__, instance_id, gpu_supp_fp16, fp16_used)

void error_on_mismatch_layout(const std::string& file,
                              int line,
                              const std::string& instance_id,
                              const std::string& layout_1_id,
                              const layout& layout_1,
                              const std::string& layout_2_id,
                              const layout& layout_2,
                              const std::string& additional_message = "");
#define CLDNN_ERROR_LAYOUT_MISMATCH(instance_id, layout_1_id, layout_1, layout_2_id, layout_2, add_msg) \
    error_on_mismatch_layout(__FILE__, __LINE__, instance_id, layout_1_id, layout_1, layout_2_id, layout_2, add_msg)

void error_on_bool(const std::string& file,
                   int line,
                   const std::string& instance_id,
                   const std::string& condition_id,
                   bool condition,
                   const std::string& additional_message = "");
#define CLDNN_ERROR_BOOL(instance_id, condition_id, condition, add_msg) \
    error_on_bool(__FILE__, __LINE__, instance_id, condition_id, condition, add_msg)

void error_on_mismatching_data_types(const std::string& file,
                                     int line,
                                     const std::string& instance_id,
                                     const std::string& data_format_1_id,
                                     data_types data_format_1,
                                     const std::string& data_format_2_id,
                                     data_types data_format_2,
                                     const std::string& additional_message = "",
                                     bool ignore_sign = false);
#define CLDNN_ERROR_DATA_TYPES_MISMATCH(instance_id,      \
                                        data_format_1_id, \
                                        data_format_1,    \
                                        data_format_2_id, \
                                        data_format_2,    \
                                        add_msg)          \
    error_on_mismatching_data_types(__FILE__,             \
                                    __LINE__,             \
                                    instance_id,          \
                                    data_format_1_id,     \
                                    data_format_1,        \
                                    data_format_2_id,     \
                                    data_format_2,        \
                                    add_msg)
#define CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(instance_id,      \
                                                    data_format_1_id, \
                                                    data_format_1,    \
                                                    data_format_2_id, \
                                                    data_format_2,    \
                                                    add_msg)          \
    error_on_mismatching_data_types(__FILE__,                         \
                                    __LINE__,                         \
                                    instance_id,                      \
                                    data_format_1_id,                 \
                                    data_format_1,                    \
                                    data_format_2_id,                 \
                                    data_format_2,                    \
                                    add_msg,                          \
                                    true)

void error_on_tensor_dims_less_than_other_tensor_dims(const std::string& file,
                                                      int line,
                                                      const std::string& instance_id,
                                                      const std::string& tensor_id,
                                                      const tensor& tens,
                                                      const std::string& tensor_to_compare_to_id,
                                                      const tensor& tens_to_compre,
                                                      const std::string& additional_message = "");
#define CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(instance_id, tensor_id, tensor_1, compare_to_id, tensor_to_compare_to, ...) \
    error_on_tensor_dims_less_than_other_tensor_dims(__FILE__,                                                         \
                                                     __LINE__,                                                         \
                                                     instance_id,                                                      \
                                                     tensor_id,                                                        \
                                                     tensor_1,                                                         \
                                                     compare_to_id,                                                    \
                                                     tensor_to_compare_to,                                             \
                                                     __VA_ARGS__)

void error_on_tensor_dims_greater_than_other_tensor_dims(const std::string& file,
                                                         int line,
                                                         const std::string& instance_id,
                                                         const std::string& tensor_id,
                                                         const tensor& tens,
                                                         const std::string& tensor_to_compare_to_id,
                                                         const tensor& tens_to_compre,
                                                         const std::string& additional_message = "");
#define CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(instance_id,                    \
                                              tensor_id,                      \
                                              tensor_1,                       \
                                              compare_to_id,                  \
                                              tensor_to_compare_to,           \
                                              ...)                            \
    error_on_tensor_dims_greater_than_other_tensor_dims(__FILE__,             \
                                                        __LINE__,             \
                                                        instance_id,          \
                                                        tensor_id,            \
                                                        tensor_1,             \
                                                        compare_to_id,        \
                                                        tensor_to_compare_to, \
                                                        __VA_ARGS__)

void error_on_tensor_dims_not_dividable_by_other_tensor_dims(const std::string& file,
                                                             int line,
                                                             const std::string& instance_id,
                                                             const std::string& tensor_id,
                                                             const tensor& tens,
                                                             const std::string& tensor_to_compare_to_id,
                                                             const tensor& tens_to_compre,
                                                             const std::string& additional_message = "");
#define CLDNN_ERROR_TENSOR_SIZES_NOT_DIVIDABLE(instance_id,                       \
                                               tensor_id,                         \
                                               tensor_1,                          \
                                               compare_to_id,                     \
                                               tensor_to_compare_to,              \
                                               ...)                               \
    error_on_tensor_dims_not_dividable_by_other_tensor_dims(__FILE__,             \
                                                            __LINE__,             \
                                                            instance_id,          \
                                                            tensor_id,            \
                                                            tensor_1,             \
                                                            compare_to_id,        \
                                                            tensor_to_compare_to, \
                                                            __VA_ARGS__)

}  // namespace cldnn
