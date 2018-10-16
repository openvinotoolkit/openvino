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
#pragma once
#include <string>
#include "program_node.h"

namespace cldnn
{

inline std::string bool_to_str(const bool condi)
{
	if (condi)
	{
		return "true";
	}
	return "false";
}

inline std::string get_extr_type(const char* str)
{
    if (!str)
    {
        return{};
    }

    while (*str && *str != '<')
    {
        ++str;
    }
    if (!*str)
    {
        return{};
    }

    auto end = str;
    while (*end && *end != '>')
    {
        ++end;
    }
    if (!*end)
    {
        return{};
    }

    return{ str + 1, end };
}

inline std::string dt_to_str(data_types dt)
{
    switch (dt)
    {
    case data_types::i8: return "i8";
    case data_types::f16: return "f16";
    case data_types::f32: return "f32";
    default:
        return "unknown (" + std::to_string(std::underlying_type_t<data_types>(dt)) + ")";
    }
}

inline std::string fmt_to_str(format fmt)
{
    switch (fmt.value)
    {
    case format::bfyx: return "bfyx";
    case format::byxf: return "byxf";
    case format::yxfb: return "yxfb";
    case format::fyxb: return "fyxb";
    case format::bs_x_bsv16: return "bs_x_bsv16";
    case format::bs_xs_xsv8_bsv8: return "bs_xs_xsv8_bsv8";
    case format::bs_xs_xsv8_bsv16: return "bs_xs_xsv8_bsv16";
    case format::os_iyx_osv16: return "os_iyx_osv16";
    case format::os_is_yx_isa8_osv8_isv4: return "os_is_yx_isa8_osv8_isv4";
    case format::byxf_af32: return "byxf_af32";
    default:
        return "unknown (" + std::to_string(fmt.value) + ")";
    }
}

}

