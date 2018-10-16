/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>
#include "api/CPP/layout.hpp"
#include "api/CPP/lrn.hpp"

namespace cldnn
{
namespace err_details
{
    void cldnn_print_error_message(std::string file, int line, std::string instance_id, std::stringstream &msg, std::string add_msg = "");
}

template <class T1, class T2>
std::ostream& operator <<(std::ostream& left, std::pair<T1, T2> const& right)
{
    left << "{ " << right.first << ", " << right.second << " }";
    return left;
}


template<typename N1, typename N2>
inline void error_on_not_equal(std::string file, int line, std::string instance_id, std::string number_id, N1 number, std::string compare_to_id, N2 number_to_compare_to, std::string additional_message = "")
{
    std::stringstream error_msg;
    {
        if (number != static_cast<decltype(number)>(number_to_compare_to))
        {
            error_msg << number_id << "(=" << number << ") is not equal to: " << compare_to_id << "(=" << number_to_compare_to << ")" << std::endl;
            err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
        }
    }
}
#define CLDNN_ERROR_NOT_EQUAL(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) error_on_not_equal(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template<typename N1, typename N2>
inline void error_on_greater_than(std::string file, int line, std::string instance_id, std::string number_id, N1 number, std::string compare_to_id, N2 number_to_compare_to, std::string additional_message = "")
{
    std::stringstream error_msg;
    if (number > static_cast<decltype(number)>(number_to_compare_to))
    {
        error_msg << number_id << "(=" << number << ") is greater than: " << compare_to_id << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_GREATER_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) error_on_greater_than(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template<typename N1, typename N2>
inline void error_on_less_than(std::string file, int line, std::string instance_id, std::string number_id, N1 number, std::string compare_to_id, N2 number_to_compare_to, std::string additional_message = "")
{
    std::stringstream error_msg;
    if (number < static_cast<decltype(number)>(number_to_compare_to))
    {
        error_msg << number_id << "(=" << number << ") is less than: " << compare_to_id << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_LESS_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) error_on_less_than(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template<typename N1, typename N2>
inline void error_on_less_or_equal_than(std::string file, int line, std::string instance_id, std::string number_id, N1 number, std::string compare_to_id, N2 number_to_compare_to, std::string additional_message = "")
{
    std::stringstream error_msg;
    if (number <= static_cast<decltype(number)>(number_to_compare_to))
    {
        error_msg << number_id << "(=" << number << ") is less or equal than: " << compare_to_id << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_LESS_OR_EQUAL_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) error_on_less_or_equal_than(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template<typename N1, typename N2>
inline void error_on_greater_or_equal_than(std::string file, int line, std::string instance_id, std::string number_id, N1 number, std::string compare_to_id, N2 number_to_compare_to, std::string additional_message = "")
{
    std::stringstream error_msg;
    if (number >= static_cast<decltype(number)>(number_to_compare_to))
    {
        error_msg << number_id << "(=" << number << ") is greater or equal than: " << compare_to_id << "(=" << number_to_compare_to << ")" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_GREATER_OR_EQUAL_THAN(instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg) error_on_greater_or_equal_than(__FILE__, __LINE__, instance_id, number_id, number, compare_to_id, number_to_compare_to, add_msg)

template<typename ptr>
inline void error_on_nullptr(std::string file, int line, std::string instance_id, std::string condition_id, ptr condition, std::string additional_message = "")
{
    std::stringstream error_msg;
    if (condition == nullptr)
    {
        error_msg << condition_id << " should not be null" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}
#define CLDNN_ERROR_NULLPTR(instance_id, condition_id, condition, add_msg) error_on_nullptr(__FILE__, __LINE__, instance_id, condition_id, condition, add_msg)

template<typename M, typename... Ms>
inline void error_on_not_proper_enum_values(std::string file, int line, std::string instance_id, std::string mode_id, M mode, std::string modes_id, Ms... modes_to_compare_to)
{
    std::stringstream error_msg;
    auto enum_value_string = [](const M& mode)->std::string {
        if (std::is_same<M, format::type>::value)
        {
            return format::traits(mode).order;
        }
        else if (std::is_same<M, cldnn_lrn_norm_region>::value)
        {
            return mode == 0 ? "cldnn_lrn_norm_region_across_channel" : "cldnn_lrn_norm_region_within_channel";
        }
        return "error during error parsing";
    };
    const std::array<const M, sizeof...(Ms)> modes{ std::forward<Ms>(modes_to_compare_to)... };
    if (std::all_of(modes.begin(), modes.end(), [&](const M& m)->int {return mode != m; }))
    {
        error_msg << mode_id << "( " << enum_value_string(mode) << " ) is incompatible with " << modes_id << ". Should be one of: ";
        for (const auto& ms : modes)
        {
            error_msg << enum_value_string(ms) << ", ";
        }
        error_msg << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg);
    }
}
#define CLDNN_ERROR_NOT_PROPER_FORMAT(instance_id, format_id, formatt, formats_ids, ...) error_on_not_proper_enum_values(__FILE__, __LINE__, instance_id, format_id, formatt, formats_ids, __VA_ARGS__)
#define CLDNN_ERROR_NOT_PROPER_LRN_NORM_REGION(instance_id, lrn_norm_region_id, lrn_norm_region, lrn_norm_region_ids, ...) error_on_not_proper_enum_values(__FILE__, __LINE__, instance_id, lrn_norm_region_id, lrn_norm_region, lrn_norm_region_ids, __VA_ARGS__)

void error_message(std::string file, int line, std::string instance_id, std::string message);
#define CLDNN_ERROR_MESSAGE(instance_id, message) error_message(__FILE__, __LINE__, instance_id, message)

void error_on_not_supported_fp16(std::string file, int line, std::string instance_id, uint8_t supp_fp16, bool fp16_used);
#define CLDNN_ERROR_NOT_SUPPORTED_FP16(instance_id, gpu_supp_fp16, fp16_used) error_on_not_supported_fp16(__FILE__, __LINE__, instance_id, gpu_supp_fp16, fp16_used)

void error_on_mismatch_layout(std::string file, int line, std::string instance_id, std::string layout_1_id, layout layout_1, std::string layout_2_id, layout layout_2, std::string additional_message = "");
#define CLDNN_ERROR_LAYOUT_MISMATCH(instance_id, layout_1_id, layout_1, layout_2_id, layout_2, add_msg) error_on_mismatch_layout(__FILE__, __LINE__, instance_id, layout_1_id, layout_1, layout_2_id, layout_2, add_msg)

void error_on_bool(std::string file, int line, std::string instance_id, std::string condition_id, bool condition, std::string additional_message = "");
#define CLDNN_ERROR_BOOL(instance_id, condition_id, condition, add_msg) error_on_bool(__FILE__, __LINE__, instance_id, condition_id, condition, add_msg)

void error_on_mismatching_data_types(std::string file, int line, std::string instance_id, std::string data_format_1_id, data_types data_format_1, std::string data_format_2_id, data_types data_format_2, std::string additional_message = "");
#define CLDNN_ERROR_DATA_TYPES_MISMATCH(instance_id, data_format_1_id, data_format_1, data_format_2_id, data_format_2, add_msg) error_on_mismatching_data_types(__FILE__, __LINE__, instance_id, data_format_1_id, data_format_1, data_format_2_id, data_format_2, add_msg)

void error_on_tensor_dims_less_than_other_tensor_dims(std::string file, int line, std::string instance_id, std::string tensor_id, tensor tens, std::string tensor_to_compare_to_id, tensor tens_to_compre, std::string additional_message = "");
#define CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(instance_id, tensor_id, tensor_1, compare_to_id, tensor_to_compare_to, ...) error_on_tensor_dims_less_than_other_tensor_dims(__FILE__, __LINE__, instance_id, tensor_id, tensor_1, compare_to_id, tensor_to_compare_to, __VA_ARGS__)

void error_on_tensor_dims_greater_than_other_tensor_dims(std::string file, int line, std::string instance_id, std::string tensor_id, tensor tens, std::string tensor_to_compare_to, tensor tens_to_compre, std::string additional_message = "");
#define CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(instance_id, tensor_id, tensor_1, compare_to_id, tensor_to_compare_to, ...) error_on_tensor_dims_greater_than_other_tensor_dims(__FILE__, __LINE__, instance_id, tensor_id, tensor_1, compare_to_id, tensor_to_compare_to, __VA_ARGS__)

}
